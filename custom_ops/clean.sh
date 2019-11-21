while read -r dir; do
    pushd $dir
    rm -rf __pycache__ *.so *.o *.pyc
    popd
done << EOF
./sampling/
./interpolation/
./grouping
EOF
