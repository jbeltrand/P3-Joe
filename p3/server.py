import numpy as np
import torch
import threading
import grpc
from concurrent import futures
import modelserver_pb2
import modelserver_pb2_grpc
import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=(('grpc.so_reuseport', 0),))
    modelserver_pb2_grpc.add_ModelServerServicer_to_server(ModelServer(), server)
    server.add_insecure_port("[::]:5440", )
    server.start()
    server.wait_for_termination()

class PredictionCache:
    def __init__(self):
        self.cache_size = 10
        self.lock = threading.Lock()
        self.coefs = None
        self.evict_order = []

    def SetCoefs(self, coefs):
        with self.lock:
            self.cache = {}
            self.coefs = coefs
            

    def Predict(self, X):
        with self.lock:
            if self.coefs is None:
                raise ValueError("Coefs must be set before making predictions.")

            X_rounded = torch.round(X * 10000) / 10000
            X_tuple = tuple(X_rounded.flatten().tolist())

            if X_tuple in self.cache:
                y = self.cache[X_tuple]
                return y, True
            else:
                y = X @ self.coefs
                self.cache[X_tuple] = y
                self.evict_order.append(X_tuple)
                if len(self.cache) > self.cache_size:
                    vic = self.evict_order.pop(0)
                    self.cache.pop(vic)
                return y, False




class ModelServer(modelserver_pb2_grpc.ModelServerServicer):
    def __init__(self):
        self.prediction_cache = PredictionCache()
    def SetCoefs(self, request, context):
        coefs = torch.tensor(request.coefs, dtype=torch.float32)
        self.prediction_cache.SetCoefs(coefs)
        return modelserver_pb2.SetCoefsResponse(error="")


    def Predict(self, request, context):
        X = request.X
        X_tensor = torch.tensor(X, dtype=torch.float32).reshape(1, -1)
        try:
            y, hit = self.prediction_cache.Predict(X_tensor)
            return modelserver_pb2.PredictResponse(y=y, hit=hit, error='')
        except Exception as e:
            error=str(e)
            y, hit = self.prediction_cache.Predict(X_tensor)
            return modelserver_pb2.PredictResponse(y=y, hit=hit, error=error)


if __name__ == "__main__":
    serve()
