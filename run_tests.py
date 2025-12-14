import subprocess
import sys
import argparse
import os
import torch
assert torch.cuda.is_available()
def main():
    parser = argparse.ArgumentParser(description='Run test suite for Dendritic project')
    parser.add_argument('--mode', choices=['unit', 'integration', 'edge', 'all'], default='all',
                        help='Test mode: unit, integration, edge, or all (default)')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('-k', '--keyword', type=str, default=None,
                        help='Only run tests which match the given substring expression')
    args = parser.parse_args()

    # Determine test markers based on mode
    markers = []
    if args.mode == 'unit':
        markers = ['unit']
    elif args.mode == 'integration':
        markers = ['integration']
    elif args.mode == 'edge':
        markers = ['edge']
    else:  # all
        markers = ['unit', 'integration', 'edge']

    # Build the pytest command using the specified Python interpreter
    python_path = r"python.exe"
    command = [python_path, '-m', 'pytest', '-v', '--capture=tee-sys', '-s']
    if args.parallel:
        command.extend(['-n', 'auto'])

    if args.coverage:
        command.extend(['--cov=dendritic', '--cov-report=term', '--cov-report=html'])

    # Add markers to the command if any are specified
    if markers:
        marker_expression = " or ".join(markers)
        command.extend(['-m', marker_expression])

    # Add keyword filter if specified
    if args.keyword:
        command.extend(['-k', args.keyword])

    # Add the tests directory
    command.append('tests/')

    # Run the command
    result = subprocess.run(command)

    # Exit with the return code of the test run
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()