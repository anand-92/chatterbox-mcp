"""
Patch torch_python.dll to fix Windows PyTorch 2.9 OverflowError

Run this with all Python processes closed:
    python patch_torch_dll.py

See: https://github.com/pytorch/pytorch/issues/162430
"""

import os
import sys
import shutil

def main():
    # Find the DLL
    site_packages = next(p for p in sys.path if 'site-packages' in p)
    dll_path = os.path.join(site_packages, 'torch', 'lib', 'torch_python.dll')

    if not os.path.exists(dll_path):
        print(f"DLL not found at: {dll_path}")
        return 1

    print(f"Found DLL: {dll_path}")

    # Read the DLL
    with open(dll_path, 'rb') as f:
        data = f.read()

    search_pattern = b'KiiiiisOl'
    replace_pattern = b'KiiiiisOK'

    if replace_pattern in data:
        print("DLL already patched!")
        return 0

    if search_pattern not in data:
        print("Pattern not found - may be different PyTorch version")
        return 1

    # Create backup
    backup_path = dll_path + '.backup'
    if not os.path.exists(backup_path):
        print(f"Creating backup: {backup_path}")
        shutil.copy2(dll_path, backup_path)

    # Patch
    print("Patching DLL...")
    patched_data = data.replace(search_pattern, replace_pattern)

    with open(dll_path, 'wb') as f:
        f.write(patched_data)

    print("DLL patched successfully!")
    print("torch.compile should now work on Windows.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
