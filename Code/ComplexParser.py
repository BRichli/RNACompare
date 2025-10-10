from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Self, TextIO, Set


class State:
    def __init__(self, id: int, type: str) -> None:
        self.type: str = type
        self.id: int =  id
        self.parents: Set[State] = set()
        self.children: Set[State] = set()
        self.within: Optional[Node] = None 

    def addParent(self, parents: State) -> None: 
        self.parents.add(parents)

    def addChildren(self, children: State) -> None: 
        self.children.add(children)

    def addNode(self, node: Node) -> None: 
        self.within = node



class MP(State): 
    def __init__(self, id: int) -> None: 
        super().__init__(id, "MP")

class ML(State): 
    def __init__(self, id: int) -> None: 
        super().__init__(id, "ML")

class MR(State): 
    def __init__(self, id: int) -> None: 
        super().__init__(id, "MR")

class IL(State): 
    def __init__(self, id: int) -> None: 
        super().__init__(id, "IL")

class IR(State): 
    def __init__(self, id: int) -> None: 
        super().__init__(id, "IR")

class D(State): 
    def __init__(self, id: int) -> None: 
        super().__init__(id, "D")

class B(State): 
    def __init__(self, id: int) -> None: 
        super().__init__(id, "B")

class S(State): 
    def __init__(self, id: int) -> None: 
        super().__init__(id, "S")

class E(State): 
    def __init__(self, id: int) -> None: 
        super().__init__(id, "E")



class StructureError(Exception):
    """Custom exception raised for structural issues in the graph."""
    def __init__(self, message: str = "A structural graph error has occurred ") -> None:
        super().__init__(message)

class LexError(Exception):
    """Custom exception raised for structural issues in the graph."""
    def __init__(self, message: str = "A structural graph error has occurred ") -> None:
        super().__init__(message)



class Node:
    def __init__(self, id: int, type: str, num_states) -> None:
        self.id: int = id
        self.type: str = type
        self.num_state: int = num_states
        self.states: Set[State] = set()
        self.parents: Set[Node] = set()
        self.children: Set[Node] = set()

    def addState(self, states: State) -> None: 
        if len(self.states) >= self.num_state:
            raise StructureError(f"Too Many States in Node: {self.id}, expected: {self.num_state}")
        self.states.add(states)

    def addParent(self, parent: Node) -> None:
        self.parents.add(parent)

    def addChild(self, child: Node) -> None:
        self.children.add(child)



class ROOT(Node): 
    def __init__(self, id: int): 
        super().__init__(id, "ROOT", 3)
        
class MATL(Node): 
    def __init__(self, id: int): 
        super().__init__(id, "MATL", 3)

class MATR(Node): 
    def __init__(self, id: int): 
        super().__init__(id, "MATR", 3)

class BIF(Node): 
    def __init__(self, id: int): 
        super().__init__(id, "BIF", 1)

class BEGL(Node): 
    def __init__(self, id: int): 
        super().__init__(id, "BEGL", 1)

class END(Node): 
    def __init__(self, id: int): 
        super().__init__(id, "END", 1)

class BEGLR(Node): 
    def __init__(self, id: int): 
        super().__init__(id, "BEGLR", 2)

class MATP(Node): 
    def __init__(self, id: int): 
        super().__init__(id, "MATP", 6)



def lexer(file: str) -> Node: 
    with open(file, 'r') as filestream: 
        flag = True
        while flag: 
            line = filestream.readline()
            if line.strip() == "CM":
                flag = False

        nodelist = []
        statedict = {}
        flag = True
        while flag: 
            line = filestream.readline()
            if not line:
                flag = False
            else:
                tokens = [token for token in line.strip().split() if token ]
                type = tokens[1]
                id = int(tokens[2])

                match type:
                    case "ROOT":
                        nodelist.append(lexNode(ROOT(id), statedict, filestream))
                    case "MATL":
                        nodelist.append(lexNode(MATL(id), statedict, filestream))
                    case "MATR":
                        nodelist.append(lexNode(MATR(id), statedict, filestream))
                    case "BIF":
                        nodelist.append(lexNode(BIF(id), statedict, filestream))
                    case "BEGL":
                        nodelist.append(lexNode(BEGL(id), statedict, filestream))
                    case "END":
                        nodelist.append(lexNode(END(id), statedict, filestream))
                    case "BEGLR":
                        nodelist.append(lexNode(BEGLR(id), statedict, filestream))
                    case "MATP":
                        nodelist.append(lexNode(MATP(id), statedict, filestream))
                    case _:
                        raise LexError(f"Un-defined node type: {type} for node: #{id}")
                    
        for node in nodelist: 
            # linking nodes
            # nodes are not listed in order
            for state in node.states:
                for child in state.children: 
                    if child.within != node: 
                        node.addChild(child.within)
                for parent in state.parents: 
                    if parent.within != node: 
                        node.addParent(parent.within)

        #at this point all states and all nodes should be linked. 
    return nodelist[0] # we only need the start node everything else is linked at this point. 
                    


def lexNode(node: Node, statedict: Dict[int, State], filestream: TextIO) -> Node: 
    #adding states
    statelist = []
    for _ in range(node.num_state):
        line = filestream.readline()
        tokens = [token for token in line.strip().split() if token ]   
        type = tokens[0]
        id = int(tokens[1])

        parents = []
        if int(tokens[2]) != -1: 
          parents = list(range(int(tokens[2])-int(tokens[3]), int(tokens[2]) + 1))

        match type:
            case "MP": 
                state = MP(id)
            case "ML":
                state = ML(id)
            case "MR":
                state = MR(id)
            case "IL":
                state = IL(id)
            case "IR":
                state = IR(id)
            case "D":
                state = D(id)
            case "B":
                state = B(id)
            case "S":
                state = S(id)
            case "E":
                state = E(id)
            case _:
                raise LexError(f"Un-defined state type: #{type} for node: #{id}")
        
        state.addNode(node)
        statedict[id] = state
        for item in parents:
            #linking states we can rely on the assumption that all states are listed in order. as specified by the infernal documentation
            state.addParent(statedict[item])
            statedict[item].addChildren(state)

        node.addState(state)
    
    return node