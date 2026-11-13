"""
EQA Generator for PGI-ALFRED Environment

Main class that manages question generation and storage.
Auto-discovers and registers all question generator functions.
"""

import os
import json
import random
import inspect
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from datetime import datetime

from pgibench.envs.pgi_alfred.eqa.thor_tracker import ThorTracker
from pgibench.envs.pgi_alfred.eqa.utils import (validate_question_dict,
                                                shuffle_options)


class EQAGenerator:
    """
    Main generator class for EQA questions.

    Auto-discovers question generator functions and provides
    interfaces for random question generation and batch export.
    """

    def __init__(self, tracker: ThorTracker, constants_module):
        """
        Initialize the EQA Generator.

        Args:
            tracker: ThorTracker instance with spatiotemporal data
            constants_module: Module containing constants (e.g., constants.py)
        """
        self.tracker = tracker
        self.constants = constants_module

        # Registered generator functions: {question_id: generator_func}
        self.generators: Dict[str, Callable] = {}

        # Generated questions storage
        self.generated_questions: List[Dict[str, Any]] = []

        # Question sequence counter for unique IDs
        self.question_counter = 0

        # Auto-discover and register generators
        self._discover_generators()

    def _discover_generators(self):
        """
        Auto-discover all question generator functions in the generators/ directory.
        """
        # Get the directory containing this module
        eqa_dir = Path(__file__).parent
        generators_dir = eqa_dir / "generators"

        if not generators_dir.exists():
            return

        # Import all Python modules in generators directory
        for module_file in generators_dir.glob("*.py"):
            if module_file.name.startswith("__"):
                continue

            # Import the module
            module_name = f"pgibench.envs.pgi_alfred.eqa.generators.{module_file.stem}"

            try:
                # Import using importlib to avoid circular imports
                import importlib
                module = importlib.import_module(module_name)

                # Find all generator functions in the module
                for name, obj in inspect.getmembers(module,
                                                    inspect.isfunction):
                    # Check if function has the required generator attributes
                    if hasattr(obj, 'is_eqa_generator'):
                        question_id = getattr(obj, 'question_id')
                        if question_id:
                            self.generators[question_id] = obj
            except ImportError as e:
                # Silently skip modules that can't be imported
                # (may have missing dependencies)
                pass

    def register_generator(self, question_id: str, func: Callable):
        """
        Manually register a generator function.

        Args:
            question_id: Unique identifier for the question type
            func: Generator function
        """
        self.generators[question_id] = func

    def generate_random_question(
            self,
            timestep: Optional[int] = None,
            seed: Optional[int] = None,
            shuffle_opts: bool = False) -> Optional[Dict[str, Any]]:
        """
        Generate a random question from a random generator.

        Args:
            timestep: Timestep to use for question (uses tracker.current_timestep if None)
            seed: Random seed for reproducibility
            shuffle_opts: Whether to shuffle option order

        Returns:
            Question dictionary or None if generation failed
        """
        if not self.generators:
            return None

        # Use provided timestep or current tracker timestep
        if timestep is None:
            timestep = self.tracker.current_timestep

        # Create random instance
        rng = random.Random(seed)

        # Try to generate a question (retry if generator returns None)
        max_attempts = min(len(self.generators), 5)
        generators_list = list(self.generators.values())

        for _ in range(max_attempts):
            generator = rng.choice(generators_list)
            try:
                question = generator(tracker=self.tracker,
                                     constants=self.constants,
                                     timestep=timestep,
                                     rng=rng)

                if question is not None:
                    # Validate question structure
                    if not validate_question_dict(question):
                        continue

                    # Add unique sequence ID
                    self.question_counter += 1
                    base_id = question['metadata'].get('question_type',
                                                       'unknown')
                    question['id'] = f"{base_id}_{self.question_counter:03d}"

                    # Shuffle options if requested
                    if shuffle_opts and question['metadata'][
                            'question_type'] == 'single_choice':
                        options, label_mapping = shuffle_options(
                            question['options'], rng)
                        question['options'] = options
                        question['correct_answer'] = label_mapping[
                            question['correct_answer']]

                    # Store and return
                    self.generated_questions.append(question)
                    return question

            except Exception as e:
                # Silently skip generators that raise exceptions
                continue

        return None

    def generate_batch_questions(
            self,
            num_questions: int,
            specific: bool = False,
            seed: Optional[int] = None,
            shuffle_opts: bool = False) -> List[Dict[str, Any]]:
        """
        Generate a batch of random questions.

        Args:
            num_questions: Number of questions to generate
            seed: Random seed for reproducibility
            shuffle_opts: Whether to shuffle option order

        Returns:
            List of generated question dictionaries
        """
        # rng = random.Random(seed)
        questions = []
        if specific:
            # Generate specific questions for each registered generator
            for question_id in self.get_registered_generators():
                _num, _attempts, _max_attempts = 0, 0, 10 * num_questions
                while _num < num_questions:
                    _attempts += 1
                    q_seed = None if seed is None else seed + len(questions)
                    question = self.generate_specific_question(
                        question_id=question_id,
                        timestep=None,
                        seed=q_seed,
                        shuffle_opts=shuffle_opts)
                    if question:
                        questions.append(question)
                        _num += 1
                    if _attempts > _max_attempts:
                        break
        else:
            _attempts, _max_attempts = 0, 10 * num_questions
            while len(questions) < num_questions:
                # Use different seed for each question
                _attempts += 1
                q_seed = None if seed is None else seed + _attempts
                question = self.generate_random_question(
                    timestep=None,  # Use current timestep
                    seed=q_seed,
                    shuffle_opts=shuffle_opts)
                if question:
                    questions.append(question)
                if _attempts > _max_attempts:
                    break

        if question is not None and len(question) > 0:
            random.shuffle(questions)

        return questions

    def generate_specific_question(
            self,
            question_id: str,
            timestep: Optional[int] = None,
            seed: Optional[int] = None,
            shuffle_opts: bool = False) -> Optional[Dict[str, Any]]:
        """
        Generate a question of a specific type.

        Args:
            question_id: ID of the question type to generate
            timestep: Timestep to use (uses current if None)
            seed: Random seed for reproducibility
            shuffle_opts: Whether to shuffle option order

        Returns:
            Question dictionary or None if generation failed
        """
        if question_id not in self.generators:
            return None

        generator = self.generators[question_id]
        rng = random.Random(seed)

        if timestep is None:
            timestep = self.tracker.current_timestep

        try:
            question = generator(tracker=self.tracker,
                                 constants=self.constants,
                                 timestep=timestep,
                                 rng=rng)

            if question is not None and validate_question_dict(question):
                # Add unique sequence ID
                self.question_counter += 1
                question['id'] = f"{question_id}_{self.question_counter:03d}"

                # Shuffle options if requested
                if shuffle_opts and question['metadata'][
                        'question_type'] == 'single_choice':
                    options, label_mapping = shuffle_options(
                        question['options'], rng)
                    question['options'] = options
                    question['correct_answer'] = label_mapping[
                        question['correct_answer']]

                self.generated_questions.append(question)
                return question

        except Exception:
            pass

        return None

    def save_to_json(self,
                     filepath: str,
                     episode_metadata: Optional[Dict] = None):
        """
        Save all generated questions to a JSON file.

        Args:
            filepath: Path to save the JSON file
            episode_metadata: Optional episode-level metadata to include
        """
        # Prepare output data
        output_data = {
            "generated_at": datetime.now().isoformat(),
            "episode_metadata": episode_metadata or {},
            "questions": self.generated_questions,
            "summary": {
                "total_questions": len(self.generated_questions),
                "question_types": self._get_question_type_summary(),
                "timesteps_covered": self._get_timestep_coverage()
            }
        }

        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        return output_data

    def _get_question_type_summary(self) -> Dict[str, int]:
        """Get summary of question types generated."""
        summary = {}
        for q in self.generated_questions:
            qtype = q.get('metadata', {}).get('question_type', 'unknown')
            summary[qtype] = summary.get(qtype, 0) + 1
        return summary

    def _get_timestep_coverage(self) -> List[int]:
        """Get list of timesteps covered by generated questions."""
        timesteps = set()
        for q in self.generated_questions:
            ts = q.get('metadata', {}).get('timestep')
            if ts is not None:
                timesteps.add(ts)
        return sorted(list(timesteps))

    def clear_questions(self):
        """Clear all generated questions."""
        self.generated_questions.clear()
        self.question_counter = 0

    def get_registered_generators(self) -> List[str]:
        """Get list of registered question type IDs."""
        return list(self.generators.keys())

    def get_question_count(self) -> int:
        """Get number of questions generated so far."""
        return len(self.generated_questions)
