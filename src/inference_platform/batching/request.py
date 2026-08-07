from dataclasses import dataclass
import re
from time import perf_counter
from uuid import uuid1
import uuid

import torch
from torch._refs import to

@dataclass

class InferenceRequest:
    input_tensor = torch.Tensor
    request_id:  str
    arrival_time: float

    @classmethod
    def create (cls, input_tensor: torch.tensor) -> "InferenceRequest":
        return cls(
            input_tensor = input_tensor,
            request_id = str(uuid1()),
            arrival_time  = perf_counter(),
        )



"""

Tiny Eg
class Person:
    def __init__(self, name: str) -> None:
        self.name = name

    def introduce(self) -> None:
        print(f"My name is {self.name}")

    @classmethod
    def create_anonymous(cls):
        return cls("Anonymous")

person_1 = Person("Anand")
person_1.introduce()

person_2 = Person.create_anonymous()
person_2.introduce()        

Person("Anand")
    ↓
person_1 is created
    ↓
person_1.introduce()
    ↓
self = person_1

Then:

Person.create_anonymous()
    ↓
cls = Person
    ↓
Person("Anonymous")
    ↓
person_2 is created

The key distinction is:

Instance method:
needs an existing object
uses self

Class method:
works from the class
uses cls
can create new objects
"""