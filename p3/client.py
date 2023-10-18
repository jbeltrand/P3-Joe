import grpc
import sys
import csv
from threading import Thread, Lock
import modelserver_pb2
import modelserver_pb2_grpc

class ClientThread(Thread):
    def __init__(self, stub, coef, csv_file):
        super(ClientThread, self).__init__()
        self.stub = stub
        self.coef = coef
        self.csv_file = csv_file
        self.hits = 0
        self.misses = 0

    def run(self):
        with open(self.csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                input_data = [float(val) for val in row]
                response = self.stub.Predict(modelserver_pb2.PredictRequest(X=input_data))
                if response.hit == True:
                    self.hits += 1
                if response.hit == False:
                    self.misses += 1

def main():

    port = int(sys.argv[1])
    coef = list(map(float, sys.argv[2].split(',')))

    channel = grpc.insecure_channel(f'localhost:{port}')
    stub = modelserver_pb2_grpc.ModelServerStub(channel)

    threads = []
    for csv_file in sys.argv[3:]:
        thread = ClientThread(stub, coef, csv_file)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    total_hits = sum(thread.hits for thread in threads)
    total_misses = sum(thread.misses for thread in threads)
    total_attempts = total_hits + total_misses
    overall_hit_rate = total_hits / total_attempts

    print(overall_hit_rate)

if __name__ == "__main__":
    main()
