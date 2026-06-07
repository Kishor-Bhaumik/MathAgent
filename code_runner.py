import os
import sys
import tempfile
import subprocess


def execute_python(code: str, timeout: int = 30) -> dict:
    """
    Execute Python code in a subprocess.

    subprocess ব্যবহার করার কারণ:
    - lsoda/odeint infinite loop হলে kill করা যায়
    - stderr (warnings) আলাদা capture হয়
    - main process কখনো halt হয় না

    Returns:
        {
          'stdout'   : str,   # print() output
          'stderr'   : str,   # lsoda warnings, deprecation notices
          'error'    : str|None,  # None = success, str = crash message
          'timed_out': bool
        }
    """
    # কোডটা temp file-এ লেখো
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False, encoding='utf-8'
    )
    try:
        # numpy, scipy, sympy আগে থেকে import করে দাও
        preamble = (
            "import numpy as np\n"
            "import numpy\n"
            "import scipy\n"
            "import sympy\n"
            "import sympy as sp\n"
        )
        tmp.write(preamble + code)
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        # returncode != 0 মানে crash
        if result.returncode != 0:
            return {
                'stdout':    stdout,
                'stderr':    stderr,
                'error':     stderr or f"Process exited with code {result.returncode}",
                'timed_out': False,
            }

        return {
            'stdout':    stdout,
            'stderr':    stderr,   # lsoda warning এখানে থাকবে
            'error':     None,
            'timed_out': False,
        }

    except subprocess.TimeoutExpired:
        return {
            'stdout':    '',
            'stderr':    '',
            'error':     f"Execution timed out after {timeout} seconds.",
            'timed_out': True,
        }
    finally:
        os.unlink(tmp_path)