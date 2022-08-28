from multiprocessing import Process,Queue

if __name__ == "__main__":

    tobeQueue = Queue()

    for i in range(1,10000):
        tobeQueue.put(i)

    for i in range(1,10000):
        think = tobeQueue.get() #remove all 9999 items, allow it to die.
        print(think)

