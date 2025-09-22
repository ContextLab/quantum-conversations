#!/usr/bin/env python3
"""
Repository cleanup script with real file operations.
NO MOCKS - actual filesystem operations with proper error handling.
"""

import os
import sys
import shutil
import hashlib
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import argparse

class RepositoryCleanup:
    """Clean up and organize the quantum-conversations repository."""

    def __init__(self, base_path: Optional[Path] = None, dry_run: bool = False, verbose: bool = False):
        """
        Initialize cleanup with optional dry-run mode.

        Args:
            base_path: Base path for the code directory
            dry_run: If True, only report what would be done without making changes
            verbose: If True, print detailed progress
        """
        self.dry_run = dry_run
        self.verbose = verbose

        # Determine base path
        if base_path:
            self.base_path = Path(base_path)
        else:
            # Find the code directory
            current = Path(__file__).parent.parent
            if current.name == 'code':
                self.base_path = current
            else:
                self.base_path = current / 'code'

        self.project_root = self.base_path.parent
        self.log = []
        self.stats = {
            'files_moved': 0,
            'files_deleted': 0,
            'directories_moved': 0,
            'directories_deleted': 0,
            'bytes_cleaned': 0,
            'duplicates_found': 0
        }

    def log_action(self, action: str, detail: str = ""):
        """Log an action with optional detail."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {action}"
        if detail:
            entry += f": {detail}"
        self.log.append(entry)
        if self.verbose:
            print(entry)

    def _hash_file(self, filepath: Path) -> str:
        """Calculate SHA256 hash of a file for deduplication."""
        hasher = hashlib.sha256()
        try:
            with open(filepath, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            self.log_action(f"Error hashing {filepath}", str(e))
            return ""

    def _get_file_size(self, path: Path) -> int:
        """Get size of file or directory in bytes."""
        if path.is_file():
            return path.stat().st_size
        elif path.is_dir():
            total = 0
            for item in path.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
            return total
        return 0

    def move_outputs_to_derivatives(self):
        """Move all output files and directories to data/derivatives."""
        derivatives_path = self.project_root / 'data' / 'derivatives'

        # Create derivatives directory if needed
        if not self.dry_run:
            derivatives_path.mkdir(parents=True, exist_ok=True)

        self.log_action("Moving outputs to derivatives", str(derivatives_path))

        # Output directories to move
        output_dirs = [
            'bumplot_final',
            'bumplot_final_demos',
            'bumplot_improved',
            'dual_probability_demo',
            'enhanced_bumplot_test',
            'test_outputs',
            'demo_outputs'
        ]

        for dir_name in output_dirs:
            src = self.base_path / dir_name
            if src.exists():
                dest = derivatives_path / dir_name
                size = self._get_file_size(src)

                if not self.dry_run:
                    self._smart_move(src, dest)

                self.log_action(f"{'Would move' if self.dry_run else 'Moved'} directory",
                              f"{src.relative_to(self.project_root)} -> {dest.relative_to(self.project_root)} ({size:,} bytes)")
                self.stats['directories_moved'] += 1
                self.stats['bytes_cleaned'] += size

        # Move individual output files (PNG, PKL, PDF)
        patterns = ['*.png', '*.pkl', '*.pdf']
        for pattern in patterns:
            for file in self.base_path.glob(pattern):
                # Skip files in subdirectories
                if file.parent != self.base_path:
                    continue

                # Determine destination subdirectory based on file prefix
                if file.name.startswith('test_'):
                    subdir = 'test_outputs'
                elif file.name.startswith('demo_'):
                    subdir = 'demo_outputs'
                elif file.name.startswith('verify_'):
                    subdir = 'verification_outputs'
                else:
                    subdir = 'misc_outputs'

                dest_dir = derivatives_path / subdir
                if not self.dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)

                dest = dest_dir / file.name
                size = self._get_file_size(file)

                if not self.dry_run:
                    shutil.move(str(file), str(dest))

                self.log_action(f"{'Would move' if self.dry_run else 'Moved'} file",
                              f"{file.name} -> {dest.relative_to(self.project_root)} ({size:,} bytes)")
                self.stats['files_moved'] += 1
                self.stats['bytes_cleaned'] += size

    def _smart_move(self, src: Path, dest: Path):
        """Move with duplicate detection and handling."""
        if dest.exists():
            # Handle directory merging
            if src.is_dir() and dest.is_dir():
                for item in src.iterdir():
                    item_dest = dest / item.name
                    self._smart_move(item, item_dest)
                # Remove empty source directory
                if not any(src.iterdir()):
                    src.rmdir()
            # Handle file collision
            elif src.is_file() and dest.is_file():
                src_hash = self._hash_file(src)
                dest_hash = self._hash_file(dest)

                if src_hash == dest_hash:
                    # Files are identical, just remove source
                    src.unlink()
                    self.stats['duplicates_found'] += 1
                    self.log_action("Removed duplicate", str(src.relative_to(self.project_root)))
                else:
                    # Files differ, keep both with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_name = f"{dest.stem}_{timestamp}{dest.suffix}"
                    new_dest = dest.parent / new_name
                    shutil.move(str(src), str(new_dest))
                    self.log_action("Renamed to avoid collision", f"{src.name} -> {new_name}")
        else:
            # No collision, simple move
            shutil.move(str(src), str(dest))

    def clean_cache_files(self):
        """Remove Python cache files and directories."""
        self.log_action("Cleaning cache files")

        # Patterns to clean
        cache_patterns = [
            '**/__pycache__',
            '**/*.pyc',
            '**/*.pyo',
            '**/*.pyd',
            '**/.pytest_cache',
            '**/.coverage',
            '**/*.coverage',
            '**/.DS_Store',
            '**/Thumbs.db'
        ]

        for pattern in cache_patterns:
            for path in self.base_path.rglob(pattern.replace('**/', '')):
                size = self._get_file_size(path)

                if not self.dry_run:
                    if path.is_dir():
                        shutil.rmtree(path)
                        self.stats['directories_deleted'] += 1
                    else:
                        path.unlink()
                        self.stats['files_deleted'] += 1

                self.log_action(f"{'Would remove' if self.dry_run else 'Removed'} cache",
                              f"{path.relative_to(self.project_root)} ({size:,} bytes)")
                self.stats['bytes_cleaned'] += size

    def organize_demo_scripts(self):
        """Move demo scripts to examples directory."""
        examples_path = self.base_path / 'examples'

        if not self.dry_run:
            examples_path.mkdir(exist_ok=True)

        self.log_action("Organizing demo scripts", str(examples_path))

        # Find all demo scripts
        demo_scripts = list(self.base_path.glob('demo_*.py'))
        demo_scripts.extend(self.base_path.glob('*_demo.py'))
        demo_scripts.extend(self.base_path.glob('generate_*.py'))
        demo_scripts.extend(self.base_path.glob('run_*.py'))
        demo_scripts.extend(self.base_path.glob('create_*.py'))
        demo_scripts.extend(self.base_path.glob('visualize_*.py'))
        demo_scripts.extend(self.base_path.glob('inspect_*.py'))
        demo_scripts.extend(self.base_path.glob('execute_*.py'))

        # Remove duplicates
        demo_scripts = list(set(demo_scripts))

        # Analyze similarity and consolidate
        script_groups = self._analyze_script_similarity(demo_scripts)

        for group_name, scripts in script_groups.items():
            if len(scripts) > 1:
                # Multiple similar scripts - keep the most recent/complete
                scripts.sort(key=lambda x: (x.stat().st_size, x.stat().st_mtime), reverse=True)
                primary = scripts[0]

                # Move primary to examples
                dest = examples_path / primary.name
                if not self.dry_run:
                    if not dest.exists():
                        shutil.move(str(primary), str(dest))

                self.log_action(f"{'Would move' if self.dry_run else 'Moved'} primary script",
                              f"{primary.name} -> examples/")
                self.stats['files_moved'] += 1

                # Log duplicates for review
                for duplicate in scripts[1:]:
                    self.log_action("Found potential duplicate",
                                  f"{duplicate.name} (similar to {primary.name})")
                    self.stats['duplicates_found'] += 1
            elif scripts:
                # Single script, just move
                script = scripts[0]
                dest = examples_path / script.name
                if not self.dry_run:
                    if not dest.exists():
                        shutil.move(str(script), str(dest))

                self.log_action(f"{'Would move' if self.dry_run else 'Moved'} script",
                              f"{script.name} -> examples/")
                self.stats['files_moved'] += 1

    def _analyze_script_similarity(self, scripts: List[Path]) -> Dict[str, List[Path]]:
        """Group similar scripts based on content analysis."""
        groups = defaultdict(list)

        for script in scripts:
            # Simple grouping by prefix patterns
            name = script.stem

            if 'bumplot' in name:
                group = 'bumplot_demos'
            elif '1000_particle' in name:
                group = '1000_particle_demos'
            elif '100_particle' in name or '200_particle' in name:
                group = 'small_particle_demos'
            elif 'tensor' in name:
                group = 'tensor_demos'
            elif 'visualization' in name or 'visualize' in name:
                group = 'visualization_tools'
            elif 'notebook' in name:
                group = 'notebook_tools'
            elif 'test' in name:
                group = 'test_scripts'
            else:
                group = name  # Unique group

            groups[group].append(script)

        return dict(groups)

    def move_test_files(self):
        """Move misplaced test files to tests directory."""
        tests_path = self.base_path / 'tests'

        self.log_action("Moving test files to tests/")

        # Find test files in wrong location
        test_files = []
        for pattern in ['test_*.py', '*_test.py']:
            for file in self.base_path.glob(pattern):
                # Only files in the main code directory
                if file.parent == self.base_path:
                    test_files.append(file)

        for test_file in test_files:
            dest = tests_path / test_file.name

            if not self.dry_run:
                if not dest.exists():
                    shutil.move(str(test_file), str(dest))
                else:
                    # File exists, check if identical
                    if self._hash_file(test_file) == self._hash_file(dest):
                        test_file.unlink()
                        self.log_action("Removed duplicate test", test_file.name)
                    else:
                        # Keep both with different names
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        new_dest = tests_path / f"{test_file.stem}_{timestamp}{test_file.suffix}"
                        shutil.move(str(test_file), str(new_dest))
                        self.log_action("Moved with rename", f"{test_file.name} -> {new_dest.name}")

            self.log_action(f"{'Would move' if self.dry_run else 'Moved'} test file",
                          f"{test_file.name} -> tests/")
            self.stats['files_moved'] += 1

    def update_gitignore(self):
        """Update .gitignore with comprehensive patterns."""
        gitignore_path = self.project_root / '.gitignore'

        self.log_action("Updating .gitignore")

        # Read existing .gitignore
        existing_lines = set()
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                existing_lines = set(line.strip() for line in f if line.strip() and not line.startswith('#'))

        # Required patterns
        required_patterns = [
            # Python
            '__pycache__/',
            '*.py[cod]',
            '*$py.class',
            '*.so',
            '.Python',
            'env/',
            'venv/',
            'ENV/',
            '.venv/',

            # Testing
            '.coverage',
            '.pytest_cache/',
            'htmlcov/',
            '*.coverage',
            '.hypothesis/',

            # Outputs in code directory
            'code/*.png',
            'code/*.pdf',
            'code/*.pkl',
            'code/output/',
            'code/outputs/',
            'code/results/',
            'code/*_output/',
            'code/*_outputs/',
            'code/bumplot_*/',
            'code/test_*/',
            'code/demo_*/',

            # IDE
            '.vscode/',
            '.idea/',
            '*.swp',
            '*.swo',
            '*~',

            # OS
            '.DS_Store',
            'Thumbs.db',
            'desktop.ini',

            # Temporary
            'tmp/',
            'temp/',
            '*.tmp',
            '*.bak',

            # Jupyter
            '.ipynb_checkpoints/',

            # Model cache
            '*.bin',
            '*.safetensors',
            'model_cache/',
        ]

        # Add missing patterns
        patterns_to_add = []
        for pattern in required_patterns:
            if pattern not in existing_lines:
                patterns_to_add.append(pattern)

        if patterns_to_add and not self.dry_run:
            with open(gitignore_path, 'a') as f:
                f.write('\n# Repository cleanup patterns\n')
                for pattern in patterns_to_add:
                    f.write(f'{pattern}\n')

        self.log_action(f"{'Would add' if self.dry_run else 'Added'} {len(patterns_to_add)} patterns to .gitignore")

    def generate_report(self) -> str:
        """Generate a detailed cleanup report."""
        report = []
        report.append("=" * 60)
        report.append("REPOSITORY CLEANUP REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Mode: {'DRY RUN' if self.dry_run else 'EXECUTED'}")
        report.append("")

        report.append("STATISTICS:")
        report.append("-" * 40)
        report.append(f"Files moved: {self.stats['files_moved']}")
        report.append(f"Files deleted: {self.stats['files_deleted']}")
        report.append(f"Directories moved: {self.stats['directories_moved']}")
        report.append(f"Directories deleted: {self.stats['directories_deleted']}")
        report.append(f"Duplicates found: {self.stats['duplicates_found']}")
        report.append(f"Total bytes cleaned: {self.stats['bytes_cleaned']:,}")
        report.append(f"Space saved: {self.stats['bytes_cleaned'] / (1024*1024):.2f} MB")
        report.append("")

        if self.log:
            report.append("DETAILED LOG:")
            report.append("-" * 40)
            for entry in self.log[-50:]:  # Last 50 entries
                report.append(entry)

        report.append("")
        report.append("=" * 60)

        return '\n'.join(report)

    def run(self):
        """Execute the complete cleanup process."""
        print(f"Starting repository cleanup (dry_run={self.dry_run})...")
        print(f"Base path: {self.base_path}")
        print(f"Project root: {self.project_root}")
        print("")

        # Phase 1: Move outputs
        self.move_outputs_to_derivatives()

        # Phase 2: Clean cache files
        self.clean_cache_files()

        # Phase 3: Organize demos
        self.organize_demo_scripts()

        # Phase 4: Move test files
        self.move_test_files()

        # Phase 5: Update .gitignore
        self.update_gitignore()

        # Generate and display report
        report = self.generate_report()
        print(report)

        # Save report
        report_path = self.base_path / 'scripts' / f'cleanup_report_{datetime.now():%Y%m%d_%H%M%S}.txt'
        if not self.dry_run:
            report_path.parent.mkdir(exist_ok=True)
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {report_path}")

        return self.stats


def main():
    """Main entry point for the cleanup script."""
    parser = argparse.ArgumentParser(description='Clean up and organize the quantum-conversations repository')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--verbose', action='store_true', help='Show detailed progress')
    parser.add_argument('--report', action='store_true', help='Only generate a report without making changes')
    parser.add_argument('--base-path', type=str, help='Base path for the code directory')

    args = parser.parse_args()

    # Report mode is essentially a verbose dry-run
    if args.report:
        args.dry_run = True
        args.verbose = True

    cleanup = RepositoryCleanup(
        base_path=args.base_path,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    stats = cleanup.run()

    # Exit with appropriate code
    if args.dry_run:
        sys.exit(0)  # Success for dry-run
    elif stats['files_moved'] > 0 or stats['files_deleted'] > 0:
        sys.exit(0)  # Success with changes
    else:
        sys.exit(1)  # No changes made (might indicate issue)


if __name__ == '__main__':
    main()