# 여기다 백엔드와 연결하기 위한 API 서버 코드를 작성합니다.
from flask import Flask, request, jsonify
from inference import run_inference