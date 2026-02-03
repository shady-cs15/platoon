from .agent import TextCraftAgent, TextCraftPromptBuilder
from .env import (
    TextCraftCodeExecutor,
    TextCraftEnv,
    TextCraftRecursiveCodeExecutor,
    TextCraftRecursiveEnv,
    # Factory functions for synthetic recipes
    create_synth_env,
    create_synth_recursive_env,
)
from .synth_recipe_generator import SynthRecipeDatabase, generate_synth_recipes
from .synth_recipe_loader import SynthRecipeLoader, create_synth_recipe_database

# Synthetic dataset (TextCraft-Synth) - deeper hierarchies and difficulty tagging
from .synth_tasks import (
    Difficulty,
    create_synth_datasets,
    get_synth_task,
    get_synth_task_ids,
    get_synth_task_ids_by_difficulty,
    load_synth_task_from_disk,
)
from .tasks import create_textcraft_datasets, get_task, get_task_ids, load_task_from_disk

__all__ = [
    # Original TextCraft (Minecraft-based)
    "TextCraftAgent",
    "TextCraftPromptBuilder",
    "TextCraftEnv",
    "TextCraftCodeExecutor",
    "TextCraftRecursiveEnv",
    "TextCraftRecursiveCodeExecutor",
    "get_task",
    "get_task_ids",
    "load_task_from_disk",
    "create_textcraft_datasets",
    # TextCraft-Synth (Synthetic, deeper hierarchies)
    # Use these factory functions to create envs with synth recipes
    "create_synth_env",
    "create_synth_recursive_env",
    "get_synth_task",
    "get_synth_task_ids",
    "get_synth_task_ids_by_difficulty",
    "load_synth_task_from_disk",
    "create_synth_datasets",
    "Difficulty",
    "SynthRecipeLoader",
    "create_synth_recipe_database",
    "SynthRecipeDatabase",
    "generate_synth_recipes",
]
