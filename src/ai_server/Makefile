proto:
	python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. service.proto

prepare:
	python -m spacy download en_core_web_sm

test: proto
	PYTHONPATH=${PWD} pytest tests/test_grpc_services.py -v

clean:
	rm -f service_pb2.py service_pb2_grpc.py
	find . -type d -name "__pycache__" -exec rm -r {} +

.PHONY: proto test clean