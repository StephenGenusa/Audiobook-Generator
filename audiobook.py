#!/usr/bin/env python3
"""
Text-to-Speech Audiobook Generator.

A comprehensive pipeline for converting large text documents into high-quality
MP3 audiobooks using the Kokoro TTS model with configurable voice synthesis
and audio processing capabilities.

Stephen Genusa - https://www.github.com/StephenGenusa

"""

import argparse
import logging
import re
import sys
import tempfile
import wave
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from pydub import AudioSegment

from kokoro import KModel, KPipeline


class TextProcessor:
    """
    Handles text preprocessing and intelligent chunking for TTS processing.

    Implements hierarchical chunking strategy: paragraphs first, then sentences
    if token limits are exceeded, ensuring semantic coherence while respecting
    model constraints with comprehensive validation and recovery.
    """

    def __init__(self, max_tokens: int = 450) -> None:
        """
        Initialize text processor with token limit buffer.

        Args:
            max_tokens: Maximum tokens per chunk, set below 510 limit for safety.
        """
        self.max_tokens = max_tokens
        self.paragraph_pattern = re.compile(r"\n\s*\n+", re.MULTILINE)
        self.sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
        self.processed_chunks = 0
        self.dropped_chunks = 0

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize input text for TTS processing.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text with normalized whitespace and removed artifacts.
        """
        # Remove excessive whitespace while preserving paragraph breaks
        cleaned = re.sub(r"[ \t]+", " ", text)
        cleaned = re.sub(r"\n[ \t]*\n", "\n\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs using whitespace delimiters with fallback.

        Args:
            text: Input text to split.

        Returns:
            List of paragraph strings, guaranteed non-empty for valid input.
        """
        # Primary splitting by double newlines
        paragraphs = self.paragraph_pattern.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Fallback: if paragraph splitting fails, split by single newlines
        if not paragraphs and text.strip():
            logging.warning("Double newline splitting failed, using single newlines")
            paragraphs = [line.strip() for line in text.split("\n") if line.strip()]

        # Ultimate fallback: treat entire text as single paragraph
        if not paragraphs and text.strip():
            logging.warning("All splitting failed, treating as single paragraph")
            paragraphs = [text.strip()]

        logging.info(f"Split text into {len(paragraphs)} paragraphs")
        return paragraphs

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using punctuation-based heuristics with fallback.

        Args:
            text: Input text to split.

        Returns:
            List of sentence strings.
        """
        sentences = self.sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Fallback: if sentence splitting fails, return original text
        if not sentences and text.strip():
            logging.warning("Sentence splitting failed, using original text")
            sentences = [text.strip()]

        return sentences

    def estimate_tokens(
        self, text: str, pipeline: KPipeline, voice: str = "af_heart"
    ) -> int:
        """
        Conservative token estimation using character-based heuristics with pipeline validation.

        Avoids pipeline state corruption by using fallback estimation primarily,
        with pipeline validation only for critical edge cases.
        """
        if not text.strip():
            return 0

        # Primary estimation: character-based heuristic (proven reliable for English TTS)
        # Based on empirical analysis: English TTS averages 3.5-4.5 chars per token
        char_based_estimate = len(text) // 4

        # Conservative upper bound estimation for safety
        conservative_estimate = len(text.split()) + (
            len(text) // 20
        )  # words + punctuation estimate

        # Use the higher estimate for safety buffer
        estimated_tokens = max(char_based_estimate, conservative_estimate)

        # Pipeline validation only for edge cases near token limits
        if (
            estimated_tokens > self.max_tokens * 0.9
        ):  # Only check when approaching limits
            try:
                # Create isolated generator to avoid state corruption
                temp_generator = pipeline(text, voice, 1.0)
                _, ps, _ = next(temp_generator)

                if ps and len(ps) > 0:
                    pipeline_tokens = len(ps)
                    # Use pipeline result if significantly different
                    if abs(pipeline_tokens - estimated_tokens) > self.max_tokens * 0.2:
                        logging.debug(
                            f"Pipeline correction: {estimated_tokens} -> {pipeline_tokens} tokens"
                        )
                        return pipeline_tokens

            except (RuntimeError, torch.cuda.OutOfMemoryError) as gpu_error:
                logging.warning(f"GPU validation failed: {gpu_error}")
                # Explicit fallback decision

            except (StopIteration, GeneratorExit) as gen_error:
                logging.debug(f"Pipeline iteration ended: {gen_error}")
                # Expected termination

            except Exception as unexpected:
                logging.error(f"Unexpected validation error: {unexpected}")
                raise  # Don't silently continue for unknown failures

        return min(estimated_tokens, self.max_tokens)  # Enforce hard limit

    def validate_chunk_tokens(
        self, chunk: str, pipeline: KPipeline, voice: str
    ) -> bool:
        """
        Simplified chunk validation using reliable token estimation.
        """
        if not chunk.strip():
            return False

        # Use conservative character-based validation
        estimated_tokens = len(chunk) // 3  # Conservative estimate (3 chars per token)
        word_estimate = len(chunk.split()) * 1.5  # Account for punctuation tokens

        # Use the higher estimate for safety
        token_estimate = max(estimated_tokens, word_estimate)

        is_valid = token_estimate <= self.max_tokens

        if not is_valid:
            logging.debug(
                f"Chunk rejected: ~{token_estimate} tokens (>{self.max_tokens}), {len(chunk)} chars"
            )

        return is_valid

    def create_chunks(
        self, text: str, pipeline: KPipeline, voice: str = "af_heart"
    ) -> List[str]:
        """
        Create chunks with comprehensive validation and error recovery.

        Implements robust chunking algorithm:
        1. Split by paragraphs with multiple fallback strategies
        2. Validate each paragraph's token count individually
        3. Sub-split oversized paragraphs by sentences
        4. Force inclusion of oversized content rather than dropping
        5. Comprehensive logging and validation

        Args:
            text: Source text to process.
            pipeline: Kokoro pipeline for tokenization.
            voice: Voice identifier for consistent tokenization.

        Returns:
            List of validated text chunks, guaranteed to cover all input text.
        """
        cleaned_text = self.clean_text(text)

        if not cleaned_text:
            logging.error("Input text is empty after cleaning")
            return []

        paragraphs = self.split_into_paragraphs(cleaned_text)
        chunks = []
        total_input_chars = len(cleaned_text)
        processed_chars = 0

        logging.info(
            f"Processing {len(paragraphs)} paragraphs from {total_input_chars} characters"
        )

        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                logging.debug(f"Skipping empty paragraph {para_idx}")
                continue

            processed_chars += len(paragraph)
            logging.debug(f"Processing paragraph {para_idx}: {len(paragraph)} chars")

            # Conservative character-based validation (3 chars per token)
            estimated_tokens = len(paragraph) // 3
            word_estimate = (
                len(paragraph.split()) * 1.5
            )  # Account for punctuation tokens
            token_estimate = max(estimated_tokens, word_estimate)

            if token_estimate <= self.max_tokens:
                chunks.append(paragraph)
                self.processed_chunks += 1
                logging.debug(
                    f"Added paragraph {para_idx} as chunk {len(chunks)} (~{token_estimate} tokens)"
                )
            else:
                # Split oversized paragraph into sentences
                logging.info(
                    f"Splitting oversized paragraph {para_idx} (~{token_estimate} tokens, {len(paragraph)} chars)"
                )

                sentences = self.split_into_sentences(paragraph)
                current_chunk = ""

                for sent_idx, sentence in enumerate(sentences):
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    # Test adding this sentence to current chunk
                    test_chunk = (
                        f"{current_chunk} {sentence}".strip()
                        if current_chunk
                        else sentence
                    )

                    # Conservative character-based validation for reliability
                    test_chars = len(test_chunk)
                    test_token_estimate = max(
                        test_chars // 3, len(test_chunk.split()) * 1.5
                    )

                    if test_token_estimate <= self.max_tokens:
                        current_chunk = test_chunk
                        logging.debug(
                            f"Added sentence {sent_idx} to growing chunk (~{test_token_estimate} tokens)"
                        )
                    else:
                        # Current chunk is full, save it and start new one
                        if current_chunk:
                            chunks.append(current_chunk)
                            self.processed_chunks += 1
                            current_chunk_tokens = max(
                                len(current_chunk) // 3,
                                len(current_chunk.split()) * 1.5,
                            )
                            logging.debug(
                                f"Completed chunk {len(chunks)}: {len(current_chunk)} chars (~{current_chunk_tokens} tokens)"
                            )

                        # Handle sentence that might be too long for any chunk
                        sentence_chars = len(sentence)
                        sentence_token_estimate = max(
                            sentence_chars // 3, len(sentence.split()) * 1.5
                        )

                        if sentence_token_estimate > self.max_tokens:
                            logging.warning(
                                f"Force-splitting oversized sentence: {sentence_chars} chars (~{sentence_token_estimate} tokens)"
                            )
                            # Split by conservative character count as last resort
                            max_chunk_chars = (
                                self.max_tokens * 2
                            )  # 2 chars per token conservative for splitting

                            for i in range(0, sentence_chars, max_chunk_chars):
                                chunk_part = sentence[i : i + max_chunk_chars]
                                if chunk_part.strip():
                                    # Try to break at word boundaries if possible
                                    if (
                                        i + max_chunk_chars < sentence_chars
                                        and " " in chunk_part
                                    ):
                                        # Find last word boundary for cleaner split
                                        last_space = chunk_part.rfind(" ")
                                        if (
                                            last_space > max_chunk_chars // 2
                                        ):  # Only if it's not too short
                                            chunk_part = chunk_part[:last_space]
                                            # Adjust position for next iteration
                                            i = i + last_space - max_chunk_chars

                                    chunks.append(chunk_part.strip())
                                    self.processed_chunks += 1
                                    logging.debug(
                                        f"Force-split chunk: {len(chunk_part)} chars"
                                    )

                            current_chunk = ""
                        else:
                            # Start new chunk with current sentence
                            current_chunk = sentence

                # Don't forget the final chunk from this paragraph
                if current_chunk:
                    chunks.append(current_chunk)
                    self.processed_chunks += 1
                    final_chunk_tokens = max(
                        len(current_chunk) // 3, len(current_chunk.split()) * 1.5
                    )
                    logging.debug(
                        f"Final chunk from paragraph {para_idx}: {len(current_chunk)} chars (~{final_chunk_tokens} tokens)"
                    )

        # Validate coverage of input text
        total_chunk_chars = sum(len(chunk) for chunk in chunks)
        coverage_ratio = (
            total_chunk_chars / total_input_chars if total_input_chars > 0 else 0
        )

        logging.info(f"Created {len(chunks)} chunks from {len(paragraphs)} paragraphs")
        logging.info(
            f"Character coverage: {total_chunk_chars}/{total_input_chars} ({coverage_ratio:.2%})"
        )
        logging.info(
            f"Processed chunks: {self.processed_chunks}, Dropped: {self.dropped_chunks}"
        )

        # Final validation - ensure no chunks are empty
        validated_chunks = [chunk for chunk in chunks if chunk.strip()]
        if len(validated_chunks) != len(chunks):
            logging.warning(
                f"Removed {len(chunks) - len(validated_chunks)} empty chunks"
            )

        # Emergency fallback: if we somehow got no chunks, split text arbitrarily
        if not validated_chunks and cleaned_text:
            logging.error("No valid chunks created, implementing emergency chunking")
            # Split into conservative character-count chunks as last resort
            max_chunk_chars = (
                self.max_tokens * 2
            )  # Conservative 2 chars per token for safety
            for i in range(0, len(cleaned_text), max_chunk_chars):
                chunk = cleaned_text[i : i + max_chunk_chars]
                if chunk.strip():
                    validated_chunks.append(chunk.strip())
            logging.warning(
                f"Emergency chunking created {len(validated_chunks)} chunks"
            )

        # Final chunk validation with token estimates
        for i, chunk in enumerate(validated_chunks):
            estimated_tokens = max(len(chunk) // 3, len(chunk.split()) * 1.5)
            if (
                estimated_tokens > self.max_tokens * 1.2
            ):  # Allow 20% buffer for estimation errors
                logging.warning(
                    f"Chunk {i} may exceed token limit: ~{estimated_tokens} tokens, {len(chunk)} chars"
                )

        return validated_chunks


class AudioGenerator:
    """
    Handles TTS audio generation using Kokoro models with GPU/CPU fallback.

    Uses the original working Kokoro pattern but processes ALL generator yields
    to capture complete audio content.
    """

    def __init__(self, use_gpu: bool = True) -> None:
        """
        Initialize audio generator with hardware configuration.

        Args:
            use_gpu: Enable CUDA acceleration if available.
        """
        self.cuda_available = torch.cuda.is_available() and use_gpu

        # Initialize models with error handling
        try:
            self.models = {
                gpu: KModel().to("cuda" if gpu else "cpu").eval()
                for gpu in [False] + ([True] if self.cuda_available else [])
            }
        except Exception as e:
            logging.error(f"Model initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize Kokoro models: {e}")

        # Initialize pipelines for American and British English variants
        try:
            self.pipelines = {
                lang_code: KPipeline(lang_code=lang_code, model=False)
                for lang_code in "ab"
            }

            # Configure pronunciation lexicon
            self.pipelines["a"].g2p.lexicon.golds["kokoro"] = "kˈOkəɹO"
            self.pipelines["b"].g2p.lexicon.golds["kokoro"] = "kˈQkəɹQ"

        except Exception as e:
            logging.error(f"Pipeline initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize Kokoro pipelines: {e}")

        logging.info(f"AudioGenerator initialized with GPU: {self.cuda_available}")

    def generate_chunk_audio(
        self, text: str, voice: str = "af_heart", speed: float = 1.0
    ) -> Tuple[int, np.ndarray]:
        """
        Generate audio using the original working Kokoro pattern but collect ALL segments.

        This uses the exact same logic as the working Gradio demo but processes
        all generator yields to capture the complete audio for long texts.

        Args:
            text: Text chunk to synthesize.
            voice: Voice identifier for synthesis.
            speed: Speech rate multiplier.

        Returns:
            Tuple of (sample_rate, complete_audio_array).

        Raises:
            RuntimeError: If synthesis fails completely.
        """
        if not text.strip():
            raise ValueError("Cannot generate audio for empty text")

        try:
            pipeline = self.pipelines[voice[0]]
            pack = pipeline.load_voice(voice)
        except Exception as e:
            raise RuntimeError(f"Failed to load voice {voice}: {e}")

        audio_segments = []
        sample_rate = 24000
        segment_count = 0

        logging.debug(f"Generating audio for {len(text)} characters")

        try:
            # Use the exact original working pattern from Kokoro demos
            for _, ps, _ in pipeline(text, voice, speed):
                if not ps:
                    logging.debug(f"Empty phoneme sequence in segment {segment_count}")
                    segment_count += 1
                    continue

                try:
                    # Original reference selection logic
                    ref_s = pack[len(ps) - 1] if len(ps) <= len(pack) else pack[-1]

                    # Generate audio with GPU/CPU fallback using original approach
                    audio_np = None

                    if self.cuda_available:
                        try:
                            audio = self.models[True](ps, ref_s, speed)
                            audio_np = audio.cpu().detach().numpy()
                        except Exception as e:
                            logging.debug(
                                f"GPU failed for segment {segment_count}: {e}"
                            )
                            # Fallback to CPU
                            audio = self.models[False](ps, ref_s, speed)
                            audio_np = audio.detach().numpy()
                    else:
                        audio = self.models[False](ps, ref_s, speed)
                        audio_np = audio.detach().numpy()

                    # Validate and collect audio
                    if audio_np is not None and len(audio_np) > 0:
                        audio_segments.append(audio_np)
                        logging.debug(
                            f"Segment {segment_count}: {len(audio_np)} samples ({len(audio_np) / sample_rate:.2f}s)"
                        )
                    else:
                        logging.warning(f"Segment {segment_count} produced no audio")

                    segment_count += 1

                except Exception as e:
                    logging.error(f"Failed to generate segment {segment_count}: {e}")
                    segment_count += 1
                    continue

        except Exception as e:
            raise RuntimeError(f"Pipeline processing failed: {e}")

        if not audio_segments:
            raise RuntimeError(
                f"No audio generated from {segment_count} attempted segments"
            )

        # Concatenate all segments into complete audio
        if len(audio_segments) == 1:
            combined_audio = audio_segments[0]
        else:
            # Ensure consistent format before concatenation
            normalized_segments = []
            for seg in audio_segments:
                if seg.ndim > 1:
                    seg = seg.flatten()
                normalized_segments.append(seg)
            combined_audio = np.concatenate(normalized_segments, axis=0)

        duration_seconds = len(combined_audio) / sample_rate
        chars_per_second = len(text) / duration_seconds if duration_seconds > 0 else 0

        logging.info(
            f"Generated {len(audio_segments)} segments, {duration_seconds:.2f}s total, {chars_per_second:.1f} chars/sec"
        )

        return sample_rate, combined_audio


class AudioProcessor:
    """
    Handles audio file operations including WAV generation and MP3 conversion.

    Manages temporary file creation, audio concatenation with silence insertion,
    and format conversion with quality optimization.
    """

    def __init__(self, silence_duration: float = 1.5, sample_rate: int = 24000) -> None:
        """
        Initialize audio processor with timing configuration.

        Args:
            silence_duration: Seconds of silence between chunks.
            sample_rate: Audio sample rate in Hz.
        """
        self.silence_duration = silence_duration
        self.sample_rate = sample_rate
        self.silence_samples = int(silence_duration * sample_rate)

    def save_chunk_wav(
        self, audio_data: np.ndarray, chunk_index: int, temp_dir: Path
    ) -> Path:
        """
        Save audio chunk as numbered WAV file with robust format handling.

        Args:
            audio_data: Audio array data.
            chunk_index: Sequential chunk number for ordering.
            temp_dir: Temporary directory for intermediate files.

        Returns:
            Path to saved WAV file.

        Raises:
            RuntimeError: If audio data is invalid or save fails.
        """
        if audio_data is None or len(audio_data) == 0:
            raise RuntimeError(f"Invalid audio data for chunk {chunk_index}")

        # Ensure audio is properly formatted for WAV output
        try:
            # Handle different input formats and convert to int16
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                # Normalize float audio to [-1, 1] range
                audio_normalized = np.clip(audio_data, -1.0, 1.0)
                audio_int16 = (audio_normalized * 32767).astype(np.int16)
            elif audio_data.dtype == np.int16:
                audio_int16 = audio_data
            else:
                # Convert other types to float first, then to int16
                audio_float = audio_data.astype(np.float32)
                audio_normalized = np.clip(audio_float, -1.0, 1.0)
                audio_int16 = (audio_normalized * 32767).astype(np.int16)

            # Ensure mono audio (flatten if needed)
            if audio_int16.ndim > 1:
                audio_int16 = audio_int16.flatten()

        except Exception as e:
            raise RuntimeError(
                f"Audio format conversion failed for chunk {chunk_index}: {e}"
            )

        wav_path = temp_dir / f"chunk_{chunk_index:04d}.wav"

        try:
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono audio
                wav_file.setsampwidth(2)  # 16-bit samples
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            # Validate file was created successfully
            if not wav_path.exists() or wav_path.stat().st_size == 0:
                raise RuntimeError(f"WAV file creation failed or file is empty")

            # Log detailed chunk information
            duration_seconds = len(audio_int16) / self.sample_rate
            logging.debug(
                f"Saved chunk {chunk_index}: {wav_path.stat().st_size:,} bytes, {duration_seconds:.2f}s"
            )
            return wav_path

        except Exception as e:
            raise RuntimeError(f"Failed to save WAV file for chunk {chunk_index}: {e}")

    def create_silence(self) -> np.ndarray:
        """
        Generate silence array for chunk separation.

        Returns:
            Numpy array containing silence samples.
        """
        return np.zeros(self.silence_samples, dtype=np.int16)

    def concatenate_wav_files(self, wav_files: List[Path], output_path: Path) -> None:
        """
        Concatenate WAV files with silence insertion using pydub with validation.

        Args:
            wav_files: List of WAV file paths in order.
            output_path: Final concatenated WAV output path.

        Raises:
            RuntimeError: If concatenation fails.
        """
        if not wav_files:
            raise RuntimeError("No WAV files provided for concatenation")

        # Validate all input files exist and log their sizes
        missing_files = []
        total_input_duration = 0

        for i, wav_file in enumerate(wav_files):
            if not wav_file.exists():
                missing_files.append(wav_file)
            else:
                file_size = wav_file.stat().st_size
                # Estimate duration from file size (rough approximation)
                estimated_duration = file_size / (self.sample_rate * 2)  # 16-bit mono
                total_input_duration += estimated_duration
                logging.debug(
                    f"Input {i}: {file_size:,} bytes, ~{estimated_duration:.1f}s"
                )

        if missing_files:
            raise RuntimeError(f"Missing WAV files: {missing_files}")

        logging.info(
            f"Concatenating {len(wav_files)} files, estimated total: {total_input_duration:.1f}s"
        )

        try:
            # Create silence segment
            silence_ms = int(self.silence_duration * 1000)
            silence = AudioSegment.silent(duration=silence_ms)

            # Load first audio segment
            combined_audio = AudioSegment.from_wav(str(wav_files[0]))
            first_duration = len(combined_audio) / 1000.0
            logging.debug(
                f"Started concatenation with {wav_files[0]} ({first_duration:.1f}s)"
            )

            # Concatenate remaining segments with silence
            for i, wav_file in enumerate(wav_files[1:], 1):
                try:
                    audio_segment = AudioSegment.from_wav(str(wav_file))
                    segment_duration = len(audio_segment) / 1000.0
                    combined_audio += silence + audio_segment

                    logging.debug(
                        f"Added chunk {i}/{len(wav_files) - 1} ({segment_duration:.1f}s)"
                    )
                except Exception as e:
                    logging.error(f"Failed to load {wav_file}: {e}")
                    raise RuntimeError(f"Concatenation failed at file {wav_file}: {e}")

            # Export concatenated result
            combined_audio.export(str(output_path), format="wav")

            # Validate output file
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError("Concatenated file was not created or is empty")

            final_duration = len(combined_audio) / 1000.0
            silence_duration = (len(wav_files) - 1) * self.silence_duration
            expected_duration = total_input_duration + silence_duration

            logging.info(
                f"Concatenation complete: {final_duration:.1f}s (expected: ~{expected_duration:.1f}s)"
            )
            logging.info(
                f"Output: {output_path} ({output_path.stat().st_size:,} bytes)"
            )

        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            else:
                raise RuntimeError(f"Audio concatenation failed: {e}")

    def convert_to_mp3(
        self, wav_path: Path, mp3_path: Path, bitrate: str = "128k"
    ) -> None:
        """
        Convert WAV file to MP3 with specified quality and validation.

        Args:
            wav_path: Source WAV file path.
            mp3_path: Destination MP3 file path.
            bitrate: MP3 encoding bitrate.

        Raises:
            RuntimeError: If conversion fails.
        """
        if not wav_path.exists():
            raise RuntimeError(f"Source WAV file does not exist: {wav_path}")

        wav_size = wav_path.stat().st_size
        logging.info(f"Converting WAV to MP3: {wav_size:,} bytes at {bitrate}")

        try:
            audio = AudioSegment.from_wav(str(wav_path))
            duration_seconds = len(audio) / 1000.0

            audio.export(
                str(mp3_path),
                format="mp3",
                bitrate=bitrate,
                parameters=["-q:a", "2"],  # High quality VBR encoding
            )

            # Validate MP3 file was created
            if not mp3_path.exists() or mp3_path.stat().st_size == 0:
                raise RuntimeError("MP3 file was not created or is empty")

            # Log conversion details
            mp3_size = mp3_path.stat().st_size
            compression_ratio = mp3_size / wav_size if wav_size > 0 else 0

            logging.info(f"MP3 conversion successful: {mp3_path}")
            logging.info(
                f"Duration: {duration_seconds:.1f}s, Size: {wav_size:,} → {mp3_size:,} bytes ({compression_ratio:.1%})"
            )

        except Exception as e:
            raise RuntimeError(f"MP3 conversion failed: {e}")


class AudiobookGenerator:
    """
    Main orchestrator for end-to-end audiobook generation pipeline.

    Coordinates text processing, audio generation, and file management
    to convert large text documents into high-quality MP3 audiobooks with
    comprehensive error handling and progress tracking.
    """

    def __init__(
        self,
        voice: str = "af_heart",
        speed: float = 1.0,
        silence_duration: float = 1.5,
        use_gpu: bool = True,
        mp3_bitrate: str = "128k",
    ) -> None:
        """
        Initialize audiobook generator with synthesis parameters.

        Args:
            voice: TTS voice identifier.
            speed: Speech synthesis speed multiplier.
            silence_duration: Silence between chunks in seconds.
            use_gpu: Enable GPU acceleration.
            mp3_bitrate: MP3 encoding quality.
        """
        self.voice = voice
        self.speed = speed
        self.mp3_bitrate = mp3_bitrate

        try:
            self.text_processor = TextProcessor()
            self.audio_generator = AudioGenerator(use_gpu=use_gpu)
            self.audio_processor = AudioProcessor(silence_duration=silence_duration)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AudiobookGenerator: {e}")

        logging.info(f"AudiobookGenerator initialized with voice: {voice}")

    def debug_chunks(self, chunks: List[str]) -> None:
        """
        Debug helper to examine chunk content and validate processing.

        Args:
            chunks: List of text chunks to examine.
        """
        logging.info("=== CHUNK DEBUG INFORMATION ===")
        for i, chunk in enumerate(chunks[:10]):  # Show first 10 chunks
            preview = chunk.replace("\n", " ").strip()
            logging.info(f"CHUNK {i}: {len(chunk)} chars")
            logging.info(f"Preview: {preview[:150]}...")
            if i < len(chunks) - 1:
                logging.info("---")

        if len(chunks) > 10:
            logging.info(f"... and {len(chunks) - 10} more chunks")
        logging.info("=== END CHUNK DEBUG ===")

    def generate_audiobook(
        self, input_text: str, output_path: Union[str, Path]
    ) -> None:
        """
        Generate complete audiobook from input text with comprehensive validation.

        Implements full pipeline: text chunking, audio synthesis, concatenation,
        and MP3 conversion with extensive error handling, progress tracking, and
        validation at each stage to ensure reliable processing of large documents.

        Args:
            input_text: Source text content for audiobook.
            output_path: Final MP3 output file path.

        Raises:
            RuntimeError: If pipeline processing fails with detailed error context.
        """
        output_path = Path(output_path)

        if not input_text or not input_text.strip():
            raise RuntimeError("Input text is empty or contains no content")

        try:
            pipeline = self.audio_generator.pipelines[self.voice[0]]
        except KeyError:
            raise RuntimeError(f"Invalid voice identifier: {self.voice}")
        except Exception as e:
            raise RuntimeError(f"Failed to access pipeline for voice {self.voice}: {e}")

        # Phase 1: Text processing and chunking with validation
        logging.info("=== PHASE 1: TEXT PROCESSING ===")
        logging.info(f"Input text: {len(input_text)} characters")

        try:
            chunks = self.text_processor.create_chunks(input_text, pipeline, self.voice)
        except Exception as e:
            raise RuntimeError(f"Text processing failed: {e}")

        if not chunks:
            raise RuntimeError("No valid text chunks generated from input text")

        # Debug chunk information
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            self.debug_chunks(chunks)

        # Validate chunk coverage
        total_input_chars = len(input_text.replace("\n", " ").strip())
        total_chunk_chars = sum(
            len(chunk.replace("\n", " ").strip()) for chunk in chunks
        )
        coverage_ratio = (
            total_chunk_chars / total_input_chars if total_input_chars > 0 else 0
        )

        logging.info(f"Generated {len(chunks)} chunks from input")
        logging.info(
            f"Character coverage: {total_chunk_chars}/{total_input_chars} ({coverage_ratio:.1%})"
        )

        if coverage_ratio < 0.9:
            logging.warning(f"Low text coverage detected: {coverage_ratio:.1%}")

        # Phase 2: Audio generation with comprehensive error handling
        logging.info("=== PHASE 2: AUDIO GENERATION ===")

        with tempfile.TemporaryDirectory(prefix="audiobook_") as temp_dir:
            temp_path = Path(temp_dir)
            wav_files = []
            failed_chunks = []
            successful_chunks = 0
            total_audio_duration = 0

            for i, chunk in enumerate(chunks):
                chunk_preview = chunk.replace("\n", " ").strip()[:100]
                logging.info(
                    f"Processing chunk {i + 1}/{len(chunks)}: {len(chunk)} chars"
                )
                logging.debug(f"Chunk preview: {chunk_preview}...")

                try:
                    # Generate audio for this chunk
                    sample_rate, audio_data = self.audio_generator.generate_chunk_audio(
                        chunk, self.voice, self.speed
                    )

                    # Validate generated audio
                    if audio_data is None or len(audio_data) == 0:
                        raise RuntimeError("Generated audio is empty or invalid")

                    # Calculate chunk duration
                    chunk_duration = len(audio_data) / sample_rate
                    total_audio_duration += chunk_duration

                    # Save as WAV file
                    wav_path = self.audio_processor.save_chunk_wav(
                        audio_data, i, temp_path
                    )
                    wav_files.append(wav_path)
                    successful_chunks += 1

                    # Enhanced progress reporting with duration estimates
                    chars_per_second = (
                        len(chunk) / chunk_duration if chunk_duration > 0 else 0
                    )
                    logging.info(
                        f"Chunk {i + 1} complete: {chunk_duration:.1f}s, {chars_per_second:.1f} chars/sec"
                    )

                    if (i + 1) % 3 == 0 or i == len(chunks) - 1:
                        success_rate = successful_chunks / (i + 1) * 100
                        avg_duration = (
                            total_audio_duration / successful_chunks
                            if successful_chunks > 0
                            else 0
                        )
                        logging.info(
                            f"Progress: {successful_chunks}/{i + 1} chunks ({success_rate:.1f}%), {total_audio_duration:.1f}s total, {avg_duration:.1f}s avg"
                        )

                except Exception as e:
                    logging.error(f"Failed to process chunk {i}: {e}")
                    logging.error(f"Chunk content preview: {chunk_preview}")
                    failed_chunks.append((i, str(e)))

                    # Continue processing remaining chunks rather than failing completely
                    continue

            # Evaluate processing results
            if failed_chunks:
                failure_rate = len(failed_chunks) / len(chunks) * 100
                logging.warning(
                    f"Failed to process {len(failed_chunks)} chunks ({failure_rate:.1f}%)"
                )

                # Log details of failed chunks for debugging
                for chunk_idx, error in failed_chunks[:5]:  # Show first 5 failures
                    logging.warning(f"Chunk {chunk_idx} failed: {error}")

                if len(wav_files) == 0:
                    raise RuntimeError(
                        "No audio chunks generated successfully - all chunks failed"
                    )
                elif failure_rate > 50:
                    logging.error(f"High failure rate: {failure_rate:.1f}%")
                    raise RuntimeError(
                        f"Too many chunks failed ({len(failed_chunks)}/{len(chunks)})"
                    )

            # Add silence duration to total time estimate
            silence_duration = (
                len(wav_files) - 1
            ) * self.audio_processor.silence_duration
            estimated_total_duration = total_audio_duration + silence_duration

            logging.info(
                f"Audio generation complete: {len(wav_files)} chunks, {total_audio_duration:.1f}s + {silence_duration:.1f}s silence = ~{estimated_total_duration:.1f}s total"
            )

            # Phase 3: Audio concatenation
            logging.info("=== PHASE 3: AUDIO CONCATENATION ===")

            try:
                concatenated_wav = temp_path / "concatenated.wav"
                self.audio_processor.concatenate_wav_files(wav_files, concatenated_wav)
            except Exception as e:
                raise RuntimeError(f"Audio concatenation failed: {e}")

            # Phase 4: MP3 conversion
            logging.info("=== PHASE 4: MP3 CONVERSION ===")

            try:
                self.audio_processor.convert_to_mp3(
                    concatenated_wav, output_path, self.mp3_bitrate
                )
            except Exception as e:
                raise RuntimeError(f"MP3 conversion failed: {e}")

        # Final validation and comprehensive summary
        if not output_path.exists():
            raise RuntimeError("Final MP3 file was not created")

        final_size = output_path.stat().st_size
        if final_size == 0:
            raise RuntimeError("Final MP3 file is empty")

        # Calculate processing metrics
        chars_per_mb = (
            len(input_text) / (final_size / 1024 / 1024) if final_size > 0 else 0
        )
        estimated_duration_minutes = estimated_total_duration / 60

        logging.info("=== AUDIOBOOK GENERATION COMPLETE ===")
        logging.info(f"Output file: {output_path}")
        logging.info(
            f"File size: {final_size:,} bytes ({final_size / 1024 / 1024:.1f} MB)"
        )
        logging.info(
            f"Estimated duration: {estimated_duration_minutes:.1f} minutes ({estimated_total_duration:.1f}s)"
        )
        logging.info(
            f"Processing success: {successful_chunks}/{len(chunks)} chunks ({successful_chunks / len(chunks) * 100:.1f}%)"
        )
        logging.info(f"Text density: {chars_per_mb:.0f} characters per MB")

        if failed_chunks:
            logging.info(
                f"Note: {len(failed_chunks)} chunks failed but processing completed"
            )

        # Sanity check: warn if output seems too small
        expected_size_mb = (
            len(input_text) / 8000
        )  # Rough estimate: ~8KB text per 1MB audio
        if (
            final_size / 1024 / 1024 < expected_size_mb * 0.1
        ):  # Less than 10% of expected
            logging.warning(
                f"Output file may be truncated: {final_size / 1024 / 1024:.1f}MB vs expected ~{expected_size_mb:.1f}MB"
            )


def setup_logging(verbose: bool = False, suppress_kokoro_warnings: bool = True) -> None:
    """Configure comprehensive logging for pipeline execution tracking."""
    import warnings
    from transformers import logging as transformers_logging

    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # Override any existing configuration
    )

    # Filter out Kokoro G2P warnings that are informational only
    if suppress_kokoro_warnings:
        # Suppress Hugging Face Hub warnings
        transformers_logging.set_verbosity_error()

        # Suppress PyTorch warnings
        warnings.filterwarnings(
            "ignore", category=UserWarning, module="torch.nn.modules.rnn"
        )
        warnings.filterwarnings(
            "ignore", category=FutureWarning, module="torch.nn.utils.weight_norm"
        )

        # Suppress Hugging Face repo_id warnings specifically
        warnings.filterwarnings("ignore", message=".*Defaulting repo_id.*")

        class KokoroWarningFilter(logging.Filter):
            def filter(self, record):
                message = record.getMessage()
                return "words count mismatch" not in message

        # Apply filter to all existing loggers and set up for future loggers
        kokoro_filter = KokoroWarningFilter()

        # Apply to root logger
        logging.getLogger().addFilter(kokoro_filter)

        # Apply to common logger names that might be used by Kokoro
        for logger_name in ["", "kokoro", "__main__", "root"]:
            logger = logging.getLogger(logger_name)
            logger.addFilter(kokoro_filter)

        # Set up a custom logging class to auto-filter new loggers
        class FilteredLogger(logging.getLoggerClass()):
            def __init__(self, name, level=logging.NOTSET):
                super().__init__(name, level)
                if suppress_kokoro_warnings:
                    self.addFilter(kokoro_filter)

        # Apply the custom logger class
        logging.setLoggerClass(FilteredLogger)


def main() -> int:
    """
    Command-line interface for audiobook generation with comprehensive validation.

    Provides extensive argument parsing, validation, and error handling for
    production audiobook generation workflows with detailed progress reporting.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    parser = argparse.ArgumentParser(
        description="Convert text documents to MP3 audiobooks using Kokoro TTS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: python audiobook_generator.py book.txt audiobook.mp3 --voice af_bella --speed 1.2",
    )

    parser.add_argument("input_file", type=Path, help="Input text file path")

    parser.add_argument("output_file", type=Path, help="Output MP3 file path")

    parser.add_argument(
        "--voice",
        default="af_heart",
        help="TTS voice identifier (e.g., af_heart, af_bella, am_michael)",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech synthesis speed multiplier (0.5-2.0)",
    )

    parser.add_argument(
        "--silence",
        type=float,
        default=1.5,
        help="Silence duration between chunks in seconds",
    )

    parser.add_argument(
        "--bitrate",
        default="128k",
        help="MP3 encoding bitrate (e.g., 128k, 192k, 256k)",
    )

    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU-only processing (disable GPU acceleration)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging with debug information",
    )

    parser.add_argument(
        "--show-kokoro-warnings",
        action="store_true",
        help="Show Kokoro G2P phoneme mismatch warnings (normally suppressed)",
    )

    args = parser.parse_args()

    # Comprehensive argument validation
    validation_errors = []

    if not args.input_file.exists():
        validation_errors.append(f"Input file does not exist: {args.input_file}")
    elif not args.input_file.is_file():
        validation_errors.append(f"Input path is not a file: {args.input_file}")

    if not 0.5 <= args.speed <= 2.0:
        validation_errors.append("Speed must be between 0.5 and 2.0")

    if args.silence < 0:
        validation_errors.append("Silence duration must be non-negative")

    # Validate output directory exists and is writable
    output_dir = args.output_file.parent
    if not output_dir.exists():
        validation_errors.append(f"Output directory does not exist: {output_dir}")
    elif not output_dir.is_dir():
        validation_errors.append(f"Output parent path is not a directory: {output_dir}")

    # Validate bitrate format
    if not re.match(r"^\d+k$", args.bitrate):
        validation_errors.append("Bitrate must be in format like '128k', '192k', etc.")

    if validation_errors:
        for error in validation_errors:
            print(f"Error: {error}", file=sys.stderr)
        return 1

    # Setup logging and begin processing
    setup_logging(args.verbose, suppress_kokoro_warnings=not args.show_kokoro_warnings)

    try:
        # Load and validate input text
        logging.info(f"Loading input file: {args.input_file}")

        try:
            with open(args.input_file, "r", encoding="utf-8") as f:
                input_text = f.read()
        except UnicodeDecodeError:
            # Try alternative encodings
            for encoding in ["latin-1", "cp1252"]:
                try:
                    with open(args.input_file, "r", encoding=encoding) as f:
                        input_text = f.read()
                    logging.warning(f"Used {encoding} encoding for input file")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise RuntimeError(
                    "Could not decode input file with any supported encoding"
                )

        if not input_text or not input_text.strip():
            raise RuntimeError("Input file is empty or contains no readable text")

        logging.info(f"Loaded {len(input_text)} characters from input file")

        # Initialize and run audiobook generator
        logging.info("Initializing audiobook generator...")

        generator = AudiobookGenerator(
            voice=args.voice,
            speed=args.speed,
            silence_duration=args.silence,
            use_gpu=not args.cpu_only,
            mp3_bitrate=args.bitrate,
        )

        # Generate audiobook
        logging.info("Starting audiobook generation...")
        generator.generate_audiobook(input_text, args.output_file)

        print(f"\nSuccess! Generated audiobook: {args.output_file}")

        # Display final file information
        if args.output_file.exists():
            file_size = args.output_file.stat().st_size
            print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")

        return 0

    except KeyboardInterrupt:
        logging.error("Process interrupted by user")
        print("\nProcess interrupted by user", file=sys.stderr)
        return 1

    except Exception as e:
        logging.error(f"Audiobook generation failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    exit(main())
