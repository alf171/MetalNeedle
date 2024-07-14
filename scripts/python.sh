ECHO $PWD
python3 -m Needle.testing

# ulimit -c unlimited
# core_dump=$(ls ~/core_dumps/core* 2>/dev/null)

# if [ -n "$core_dump" ]; then
#     echo "Core dump detected: $core_dump. Starting lldb for debugging..."
#     lldb -c "$core_dump"
# else
#     echo "No core dump found."
# fi