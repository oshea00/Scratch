class MinHeap:
    def __init__(self):
        self.data = []

    def parent(self, i):
        return (i - 1) // 2

    def left(self, i):
        return 2 * i + 1

    def right(self, i):
        return 2 * i + 2

    def push(self, value):
        self.data.append(value)
        self._sift_up(len(self.data) - 1)

    def isEmpty(self):
        return not self.data

    def pop(self):
        if not self.data:
            raise IndexError("pop from empty heap")

        root = self.data[0]
        last = self.data.pop()

        if self.data:
            self.data[0] = last
            self._sift_down(0)

        return root

    def peek(self):
        if not self.data:
            raise IndexError("peek from empty heap")
        return self.data[0]

    def _sift_up(self, i):
        while i > 0:
            p = self.parent(i)

            if self.data[p] <= self.data[i]:
                break

            self.data[p], self.data[i] = self.data[i], self.data[p]

            i = p

    def _sift_down(self, i):
        n = len(self.data)

        while True:
            left = self.left(i)
            right = self.right(i)
            smallest = i

            if left < n and self.data[left] < self.data[smallest]:
                smallest = left

            if right < n and self.data[right] < self.data[smallest]:
                smallest = right

            if smallest == i:
                break

            self.data[i], self.data[smallest] = self.data[smallest], self.data[i]

            i = smallest


h = MinHeap()

h.push(9)
h.push(2)
h.push(1)
h.push(5)
h.push(7)
h.push(4)
while not h.isEmpty():
    print(h.pop())
print(h.isEmpty())
