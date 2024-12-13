"""
Module for a Doubly Linked List implementation.

This module consists of the DLLNode class,
a node in the context of a doubly linked list,
and the DoublyLinkedList class that
contains the required operations of a
doubly linked list.
"""


class DLLNode:
    """Node for a doubly linked list."""

    def __init__(self, data):
        """Initialize a node with data, prev and next pointers."""
        self.data = data
        self.prev = None
        self.next = None


class DoublyLinkedList:
    """Doubly Linked List class."""

    def __init__(self):
        """Initialize an empty doubly linked list."""
        self.head = None
        self._tail = None
        self._curr = None

    def add_to_head(self, data):
        """Add a node to the head of the list."""
        new_node = DLLNode(data)
        if self.is_empty():
            self.head = new_node
            self._tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self._curr = self.head

    def add_after_current(self, data):
        """Add a node after the current node."""
        if self._curr is None:
            raise IndexError("Current node is None, cannot add after it.")

        new_node = DLLNode(data)
        new_node.prev = self._curr
        new_node.next = self._curr.next

        if self._curr.next is not None:
            self._curr.next.prev = new_node
        else:
            self._tail = new_node

        self._curr.next = new_node

    def remove_from_head(self):
        """Remove the node from the head of the list."""
        if self.is_empty():
            raise IndexError("List is empty.")

        removed_data = self.head.data
        if self.head == self._tail:  # Only one node in the list
            self.head = None
            self._tail = None
        else:
            self.head = self.head.next
            self.head.prev = None
        self._curr = self.head
        return removed_data

    def remove_after_current(self):
        """Remove the node after the current node."""
        if self._curr is None or self._curr.next is None:
            raise IndexError("There is no node to remove after the current.")

        to_remove = self._curr.next
        removed_data = to_remove.data
        self._curr.next = to_remove.next

        if to_remove.next is not None:
            to_remove.next.prev = self._curr
        else:
            self._tail = self._curr  # We're removing the tail

        return removed_data

    def reset_to_head(self):
        """Reset current pointer to the head of the list."""
        self._curr = self.head

    def reset_to_tail(self):
        """Reset current pointer to the tail of the list."""
        self._curr = self._tail

    def move_forward(self):
        """Move the current pointer forward."""
        if self._curr is None or self._curr.next is None:
            raise IndexError("Current node at the tail; can't move forward.")
        self._curr = self._curr.next

    def move_backward(self):
        """Move the current pointer backward."""
        if self._curr is None or self._curr.prev is None:
            raise IndexError("Current node at the head, can't move backward.")
        self._curr = self._curr.prev

    def find(self, data):
        """Find and return the data in the list."""
        current = self.head
        while current:
            if current.data == data:
                return data
            current = current.next
        raise IndexError("Data not found in the list.")

    def remove(self, data):
        """Remove a node with the given data."""
        if self.is_empty():
            raise IndexError("List is empty, cannot remove.")

        current = self.head
        while current:
            if current.data == data:
                if current.prev:  # Not head
                    current.prev.next = current.next
                else:  # Is head
                    self.head = current.next

                if current.next:  # Not tail
                    current.next.prev = current.prev
                else:  # Is tail
                    self._tail = current.prev

                self._curr = self.head  # Reset current to head
                return data
            current = current.next
        raise IndexError("Data not found in the list.")

    @property
    def curr_data(self):
        """Return data of the current node."""
        if self._curr is None:
            raise IndexError("Current node is None.")
        return self._curr.data

    def is_empty(self):
        """Return True if the list is empty, False otherwise."""
        return self.head is None
