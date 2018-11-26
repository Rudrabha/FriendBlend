This is the repository for Team 007 for Digital Image Processing (CSE478) Project.

The major goal for this project would be to implement FriendBlend (https://web.stanford.edu/class/ee368/Project_Spring_1415/Reports/Chen_Zeng.pdf).

Team Members are - \
Rudrabha Mukhopadhyay \
Sangeeth Reddy Battu 

### To run FriendBlend

python main.py --inp1 *Path to input file 1* --inp2 *Path to input file 2* 

To specify the path to output file, 

python main.py --inp1 *Path to input file 1* --inp2 *Path to input file 2* --op *Path to output file*

The program with automatically try to find whether it should use grabcut or alpha blending. To override this, 

python main.py --inp1 *Path to input file 1* --inp2 *Path to input file 2* --technique *grabcut/alphablend*


