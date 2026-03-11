"""
Basic tests for the Jalapeno parser (minimal Filament subset)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parser import create_parser

def test_parse_import():
    parser = create_parser()
    code = 'import "primitives/core.fil";'
    tree = parser.parse(code)
    assert tree.data == "start"
    print("  OK: import statement")

def test_parse_simple_comp():
    parser = create_parser()
    code = '''
    comp Add[W]<'G: 1>(
        go: interface['G],
        left: ['G, 'G+1] W,
    ) -> (
        out: ['G+1, 'G+2] W
    ) {
    }
    '''
    tree = parser.parse(code)
    assert tree.data == "start"
    print("  OK: simple component")

def test_parse_with_constraint():
    parser = create_parser()
    code = '''
    comp Add[W]<'G: 1>(
        left: ['G, 'G+1] W,
    ) -> (
        out: ['G+1, 'G+2] W
    ) where W > 0 {
    }
    '''
    tree = parser.parse(code)
    assert tree.data == "start"
    print("  OK: component with constraint")

def test_parse_instance():
    parser = create_parser()
    code = '''
    comp Test<'G: 1>() -> () {
        add := new AddComb[32];
    }
    '''
    tree = parser.parse(code)
    assert tree.data == "start"
    print("  OK: instance")

def test_parse_invocation():
    parser = create_parser()
    code = '''
    comp Test<'G: 1>() -> () {
        add := new AddComb[32];
        result := add<'G>(left, right);
    }
    '''
    tree = parser.parse(code)
    assert tree.data == "start"
    print("  OK: invocation")

def test_parse_connect():
    parser = create_parser()
    code = '''
    comp Test<'G: 1>() -> () {
        out = result.out;
    }
    '''
    tree = parser.parse(code)
    assert tree.data == "start"
    print("  OK: connect")

def test_parse_example_file():
    parser = create_parser()
    example_path = Path(__file__).parent.parent / "examples" / "basic.fil"
    tree = parser.parse_file(str(example_path))
    assert tree.data == "start"
    print("  OK: example file basic.fil")


if __name__ == "__main__":
    print("Running parser tests...")

    tests = [
        test_parse_import,
        test_parse_simple_comp,
        test_parse_with_constraint,
        test_parse_instance,
        test_parse_invocation,
        test_parse_connect,
        test_parse_example_file,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
