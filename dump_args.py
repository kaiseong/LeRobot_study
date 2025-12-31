cat > dump_argv.py <<'PY'
import sys
for i,a in enumerate(sys.argv):
    head = a[:120]
    print(i, repr(head))
    if head and ord(head[0]) == 0x1b:
        print("  --> FIRST CHAR IS ESC (0x1b)")
    if "\x1b" in a:
        print("  --> CONTAINS ESC SOMEWHERE")
PY
