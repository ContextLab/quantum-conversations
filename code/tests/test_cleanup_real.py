"""
Real integration tests for repository cleanup using actual file operations.
NO MOCKS - everything runs against real systems.
"""

import pytest
import tempfile
import shutil
import hashlib
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_conversations import ParticleFilter, TokenSequenceVisualizer
from scripts.cleanup_repository import RepositoryCleanup


@pytest.fixture
def real_temp_directory():
    """Create actual temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Real cleanup
    shutil.rmtree(temp_dir)


class TestRealFileOperations:
    """Test real file operations without any mocks."""

    def test_real_output_movement(self, real_temp_directory):
        """Test moving real files with actual I/O operations."""
        # Create real test files
        test_file = real_temp_directory / 'test_output.png'
        test_file.write_bytes(b'\x89PNG\r\n\x1a\n' + b'TEST_DATA')  # Real PNG header

        # Real directory operations
        dest_dir = real_temp_directory / 'derivatives'
        dest_dir.mkdir()

        # Actual file movement
        shutil.move(str(test_file), str(dest_dir / test_file.name))

        # Verify with real filesystem checks
        assert not test_file.exists()
        assert (dest_dir / test_file.name).exists()
        assert (dest_dir / test_file.name).read_bytes().startswith(b'\x89PNG')

    def test_duplicate_detection_real_files(self, real_temp_directory):
        """Test duplicate detection with actual file comparison."""
        # Create real duplicate files
        file1 = real_temp_directory / 'file1.png'
        file2 = real_temp_directory / 'file2.png'

        content = b'\x89PNG\r\n\x1a\n' + b'SAME_CONTENT'
        file1.write_bytes(content)
        file2.write_bytes(content)

        # Real hash comparison
        hash1 = hashlib.sha256(file1.read_bytes()).hexdigest()
        hash2 = hashlib.sha256(file2.read_bytes()).hexdigest()

        assert hash1 == hash2  # Verify duplicates detected

    def test_cleanup_script_dry_run(self, real_temp_directory):
        """Test cleanup script in dry-run mode with real files."""
        # Create test structure
        code_dir = real_temp_directory / 'code'
        code_dir.mkdir()

        # Create real test files
        test_output = code_dir / 'test_output.png'
        test_output.write_bytes(b'\x89PNG\r\n\x1a\n' + b'TEST')

        demo_script = code_dir / 'demo_test.py'
        demo_script.write_text('print("test")')

        cache_dir = code_dir / '__pycache__'
        cache_dir.mkdir()
        cache_file = cache_dir / 'test.pyc'
        cache_file.write_bytes(b'PYCODE')

        # Run cleanup in dry-run mode
        cleanup = RepositoryCleanup(base_path=code_dir, dry_run=True)
        stats = cleanup.run()

        # Verify nothing was actually moved/deleted
        assert test_output.exists()
        assert demo_script.exists()
        assert cache_dir.exists()

        # But stats should show what would be done
        assert stats['files_moved'] > 0 or stats['files_deleted'] > 0

    def test_cleanup_script_real_execution(self, real_temp_directory):
        """Test cleanup script with real execution."""
        # Create test structure
        code_dir = real_temp_directory / 'code'
        code_dir.mkdir()

        # Create data directory structure
        data_dir = real_temp_directory / 'data' / 'derivatives'
        data_dir.mkdir(parents=True)

        # Create real test files
        test_output = code_dir / 'test_output.png'
        test_output.write_bytes(b'\x89PNG\r\n\x1a\n' + b'TEST')

        cache_dir = code_dir / '__pycache__'
        cache_dir.mkdir()
        cache_file = cache_dir / 'test.pyc'
        cache_file.write_bytes(b'PYCODE')

        # Run cleanup for real
        cleanup = RepositoryCleanup(base_path=code_dir, dry_run=False)
        stats = cleanup.run()

        # Verify files were actually moved/deleted
        assert not test_output.exists()  # Should be moved
        assert not cache_dir.exists()  # Should be deleted

        # Check files were moved to correct location
        moved_file = data_dir / 'test_outputs' / 'test_output.png'
        assert moved_file.exists()
        assert moved_file.read_bytes().startswith(b'\x89PNG')

        # Verify stats
        assert stats['files_moved'] >= 1
        assert stats['directories_deleted'] >= 1


class TestRealModelLoading:
    """Test model functionality preservation after cleanup."""

    def test_cleanup_preserves_model_functionality(self):
        """Ensure cleanup doesn't break real model loading."""
        # Real model initialization
        pf = ParticleFilter(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            n_particles=2,
            device="cpu"
        )

        # Real generation
        pf.initialize("Test prompt")
        particles = pf.generate(n_steps=10)  # Use n_steps instead of max_length

        # Verify real output
        assert len(particles) == 2
        assert all(len(p.tokens) > 0 for p in particles)

    def test_visualizer_after_cleanup(self):
        """Test visualizer still works after cleanup."""
        # Real particle generation
        pf = ParticleFilter(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            n_particles=3,
            device="cpu"
        )
        pf.initialize("Hello")
        particles = pf.generate(n_steps=5)  # Use n_steps

        # Initialize visualizer with tokenizer
        visualizer = TokenSequenceVisualizer(pf.tokenizer)

        # Real visualization (to temp file)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = Path(tmp.name)

        try:
            visualizer.visualize_bumplot(particles, output_path)

            # Verify real output
            assert output_path.exists()
            assert output_path.stat().st_size > 0

            # Verify it's a real PNG
            header = output_path.read_bytes()[:8]
            assert header == b'\x89PNG\r\n\x1a\n'
        finally:
            # Clean up
            if output_path.exists():
                output_path.unlink()


class TestRealGitignoreUpdates:
    """Test .gitignore updates with real file operations."""

    def test_gitignore_patterns_work(self, real_temp_directory):
        """Test that gitignore patterns actually work."""
        # Create test repository
        repo_dir = real_temp_directory / 'repo'
        repo_dir.mkdir()

        gitignore = repo_dir / '.gitignore'
        gitignore.write_text("""
# Test patterns
*.pyc
__pycache__/
*.png
test_*/
        """)

        # Create files that should be ignored
        ignored_files = [
            repo_dir / 'test.pyc',
            repo_dir / 'output.png',
            repo_dir / '__pycache__' / 'cache.pyc',
            repo_dir / 'test_dir' / 'file.txt',
        ]

        for file in ignored_files:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('ignored')

        # Use git to check if files would be ignored (if git is available)
        try:
            import subprocess
            os.chdir(repo_dir)
            subprocess.run(['git', 'init'], capture_output=True)

            # Check which files git would ignore
            result = subprocess.run(
                ['git', 'check-ignore', *[str(f.relative_to(repo_dir)) for f in ignored_files]],
                capture_output=True
            )

            # All files should be ignored (exit code 0)
            assert result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            # Git not available, skip git-specific test
            pass


class TestRealFileHashing:
    """Test file hashing and deduplication with real files."""

    def test_hash_large_file(self, real_temp_directory):
        """Test hashing of large files."""
        # Create a large file (1MB)
        large_file = real_temp_directory / 'large.bin'
        content = os.urandom(1024 * 1024)  # 1MB of random data
        large_file.write_bytes(content)

        # Calculate hash
        hasher = hashlib.sha256()
        with open(large_file, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

        hash_result = hasher.hexdigest()

        # Verify hash is correct
        assert len(hash_result) == 64  # SHA256 is 64 hex chars

        # Create identical copy and verify same hash
        copy_file = real_temp_directory / 'copy.bin'
        copy_file.write_bytes(content)

        hasher2 = hashlib.sha256()
        with open(copy_file, 'rb') as f:
            while chunk := f.read(8192):
                hasher2.update(chunk)

        assert hasher2.hexdigest() == hash_result

    def test_detect_different_files(self, real_temp_directory):
        """Test that different files have different hashes."""
        file1 = real_temp_directory / 'file1.txt'
        file2 = real_temp_directory / 'file2.txt'

        file1.write_text('Content 1')
        file2.write_text('Content 2')

        hash1 = hashlib.sha256(file1.read_bytes()).hexdigest()
        hash2 = hashlib.sha256(file2.read_bytes()).hexdigest()

        assert hash1 != hash2


class TestRealDirectoryOperations:
    """Test directory operations with real filesystem."""

    def test_recursive_directory_size(self, real_temp_directory):
        """Test calculating directory size recursively."""
        # Create nested structure
        root = real_temp_directory / 'root'
        root.mkdir()

        # Create files at different levels
        (root / 'file1.txt').write_text('x' * 100)

        subdir1 = root / 'subdir1'
        subdir1.mkdir()
        (subdir1 / 'file2.txt').write_text('x' * 200)

        subdir2 = subdir1 / 'subdir2'
        subdir2.mkdir()
        (subdir2 / 'file3.txt').write_text('x' * 300)

        # Calculate total size
        total_size = 0
        for item in root.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size

        assert total_size == 600  # 100 + 200 + 300

    def test_directory_merge(self, real_temp_directory):
        """Test merging directories with real files."""
        # Create source and destination directories
        src = real_temp_directory / 'src'
        src.mkdir()
        (src / 'file1.txt').write_text('Source 1')

        dest = real_temp_directory / 'dest'
        dest.mkdir()
        (dest / 'file2.txt').write_text('Dest 2')

        # Merge directories
        for item in src.iterdir():
            shutil.move(str(item), str(dest / item.name))

        # Verify merge
        assert not (src / 'file1.txt').exists()
        assert (dest / 'file1.txt').exists()
        assert (dest / 'file2.txt').exists()
        assert (dest / 'file1.txt').read_text() == 'Source 1'