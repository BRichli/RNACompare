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

More code 

 ̄function tree-edit-distance(Two input trees  ̄x and  ̄y, a pseudo-metric c.)
m ←| ̄x|, n ←| ̄y|.
d ←m ×n matrix of zeros. ⊲ di,j = dc( ̄xi,  ̄yj ).
D ←(m + 1) ×(n + 1) matrix of zeros.
⊲ Di,j = Dc(X[i, r ̄x(k)], Y [j, r ̄y (l)]).
for k ←m, . . . , 1 do
for l ←n, . . . , 1 do
DrX (k)+1,rY (l)+1 ←0. ⊲ equation 20
for i ←rX (k), . . . , k do
Di,rY (l)+1 ←Di+1,rY (l)+1 + c(xi, −). ⊲ equation 21
end for

for j ←rY (l), . . . , l do
DrX (k)+1,j ←DrX (k)+1,j+1 + c(−, yj). ⊲ equation 22
end for

for i ←rX (k), . . . , k do
for j ←rY (l), . . . , l do
if r ̄x(i) = r ̄x(k) ∧r ̄y(j) = r ̄y(l) then
Di,j ←min{Di+1,j + c(xi, −),
Di,j+1 + c(−, yj),
Di+1,j+1 + c(xi, yj )}. ⊲ equation 16
di,j ←Di,j .
else
Di,j ←min{Di+1,j + c(xi, −),
Di,j+1 + c(−, yj),
Dr ̄x(i)+1,r ̄y(j)+1 + di,j }. ⊲ equation 15
end if
end for
end for
end for
end for

return d1,1.
end function