"""
OpenCITE CLI - Command-line interface for OpenCITE.
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="OpenCITE - Multi-platform AI Discovery & Cataloging Tool"
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch the web-based GUI')
    gui_parser.add_argument('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    gui_parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    gui_parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    if args.command == 'gui':
        from open_cite.gui.app import run_gui
        run_gui(host=args.host, port=args.port, debug=args.debug)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
