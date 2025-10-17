# DNASnipFiltering
Data inspection tool for DNA sequences

## Resources


#### Link to Articles


#### Links to SCFM's


#### Links to Documentation


## Goal
#### Take in two CFG's which represent families of RNA and compare the models to determine similarity of RNA families. 


## Questions for Us

#### 1) What is the constraint that a covariance model places upon a scfg. 
#### 2) Is a covariance model identical to a Visually Pushdown Automata (IE Can we use work on both?) (Bryan)
#### 3) 


## Questions for Zhong

#### 1) Clarify that we are only trying to accomplish the same thing that HHSEARCH does (comparing two models) but we aren't using HMMS
#### 2) Is he aware of current methods to do SCFG to SCFG distance comparison (I think the answer is no), if this turns out to be too challenging may we resort to the HMM-HMM comparison.  
#### 3) Do we care about comparing the content of the rna sequences or only their secondary structures (shapes). 
#### 4) Can we instead of comparing languages, compare grammars as an approximation of language similarity. 




## Tertiary Questions

#### 



## Timeline


## TODO's 
#### 1) Meet with Zhong tomorrow (ALL) (9/23)
#### 2) Make Tex Doc (Bryan) (9/22)
#### 3) Make problem statement paragraph (Bryan) (9/22)



## Algorithm Candidates

```
Algorithm 1 An algorithm to transform a mapping M into a corresponding
edit script  ̄δM according to theorem 2.
function map-to-script(Two forests X and Y , a tree mapping M between
X and Y .)
I ←{i|∄j : (i, j) ∈M }.
J ←{j|∄i : (i, j) ∈M }.
Initialize  ̄δ as empty.
for (i, j) ∈M do ̄δ ←  ̄δ ⊕repi,yj . ⊲ replacements
end for

for i ∈I in descending order do ̄δ ←  ̄δ ⊕deli. ⊲ deletions
end for

(n, R0, . . . , Rn) ← num-descendants(Y , 0, J). ⊲ the number of
children for each inserted node
for j ∈J in ascending order do ̄δ ←  ̄δ ⊕insp(j)+1,ν( ̄yj),rj,rj+Rj , ⊲ insertions
end for

return  ̄δ.
end function

function num-descendants(A forest Y =  ̄y1, . . . ,  ̄yR, an index j, and an
index set J) ̄R ←ǫ. ̃R ←0. ⊲ The number of mapped descendants of this forest
for r ←1, . . . , R do
j ←j + 1.
(j′,  ̃Rj, . . . ,  ̃Rj′ ) ← num-descendants( ̺̄( ̄yr), j, J).
 ̄R ←  ̄R ⊕  ̃Rj , . . . ,  ̃Rj′ .
if j /∈J then ̃R ←  ̃R + 1.
else ̃R ←  ̃R +  ̃Rj .
end if

j ←j′.
end for

return (j,  ̃R ⊕  ̄R).
end function
12
```

this candidate requires a preorder traversal index of both trees, then it requires a order preserving mapping between the two trees (think galois but not quite). 

