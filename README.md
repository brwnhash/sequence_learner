# Sequence Learner
 Sequnce Learner implements the idea of one shot sequence learning using generative hebbian style learning . Core cocept is similar to Numentas HTM temproral learner but we have a different implementation ,rather than being biological correct ,we tried to have biological inspired implementation .
Learner more of tries to do memorization of sequnces like humans do .

For proof of concept there is multipication table memorization in run_tests.py .
    Multipication table of 1 to 30 

    python3 run_tests.py

    

    output:


     next table for 1 
     .....
     .......


    next table for 30 and initial values given are [30, 60, 90]
        [{'match': 120, 'score': 0.8}, {'match': 121, 'score': 0.2}],
        [{'match': 150, 'score': 0.666667}, {'match': 151, 'score': 0.166667}, {'match': 149, 'score': 0.166667}],
        [{'match': 180, 'score': 0.666667}, {'match': 181, 'score': 0.166667}, {'match': 179, 'score': 0.166667}],
        [{'match': 210, 'score': 0.8}, {'match': 209, 'score': 0.2}],
        [{'match': 240, 'score': 0.8}, {'match': 241, 'score': 0.2}],
        [{'match': 270, 'score': 0.8}, {'match': 269, 'score': 0.2}]

    initial values are starting reference values after which next possible values must be predicted .
    for table of 30 ,we have given 30,60,90 as start values .As learner has learnt multipication tables.
    next values are  120,150,180,210,240,270  . output also shows for every iteration other possible match like 
    in first row 121 is other match along with 120 .That is because we had settings where we allow nearby numbers to have
    overlapping bits .Idea of overlapping bits is similar to word embeddings.


Idea works for building word embeddings ,language modelling as well but those learners are not added here .



