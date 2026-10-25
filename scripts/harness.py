"""Playground test runner. Usage: python harness.py user_code.py tests.py

Runs the user's code, then every test_* function in tests.py against the same globals.
Prints one JSON line per test, then a summary line. Used by the app and by build_exercises.py.
"""
import contextlib
import io
import json
import sys
import time
import traceback


def main(code_path, tests_path):
    env = {"__name__": "__playground__"}
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(open(code_path).read(), "your_code.py", "exec"), env)
    except NotImplementedError:
        pass                                            # starter stubs: let the tests report it
    except Exception:
        print(json.dumps({"type": "load_error", "error": traceback.format_exc(limit=3)}))
        print(json.dumps({"type": "summary", "passed": 0, "total": 0, "stdout": out.getvalue()}))
        return
    test_env = dict(env)
    exec(compile(open(tests_path).read(), "tests.py", "exec"), test_env)
    tests = [k for k, v in test_env.items() if k.startswith("test_") and callable(v)]
    passed = 0
    for name in tests:
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(out):
                test_env[name]()
            ok, err = True, ""
        except NotImplementedError:
            ok, err = False, "not implemented yet"
        except AssertionError as e:
            ok, err = False, f"assertion failed{': ' + str(e) if str(e) else ''}"
        except Exception as e:
            ok, err = False, f"{type(e).__name__}: {e}"
        passed += ok
        print(json.dumps({"type": "test", "name": name, "ok": ok, "error": err,
                          "ms": round((time.perf_counter() - t0) * 1000, 1)}))
    print(json.dumps({"type": "summary", "passed": passed, "total": len(tests), "stdout": out.getvalue()[-4000:]}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
