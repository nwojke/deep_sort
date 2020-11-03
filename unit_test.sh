#!/usr/bin/env sh
# -*- coding: utf-8 -*-

tf="tf1.15"
if [[ $# -gt 0 ]]; then
    tf=$1
fi

# 单元测试
source env.bashrc $tf
python3 my_deep_sort_app.py --feature_type 1 --display False
