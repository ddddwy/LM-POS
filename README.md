# LM-POS

## Train
1. Single LSTM LM
```
python main.py --exp lm --rnn_num 1 --nhid 150 --lm --cuda
```

2. Single Weighted LSTM ([Token, POS Tag])
```
python main.py --exp single_weighted --rnn_num 1 --nhid 300 --cuda
```

3. Single Unweighted LSTM ([Token, POS Tag])
```
python main.py --exp single_unweighted --rnn_num 1 --nhid 300 --simple --cuda
```

4. Multiple Weighted LSTM (Token + POS Tag)
```
python main.py --exp multi_weighted --rnn_num 2 --nhid 150 --cuda
```

5. Multiple Unweighted LSTM (Token + POS Tag)
```
python main.py --exp multi_unweighted --rnn_num 2 --nhid 150 --simple --cuda
```

## Test
1. Single LSTM LM
```
python main.py --exp lm --test --lm --cuda
```

2. Single Weighted LSTM ([Token, POS Tag])
```
python main.py --exp single_weighted --test --cuda
```

3. Single Unweighted LSTM ([Token, POS Tag])
```
python main.py --exp single_unweighted --test --simple --cuda
```

4. Multiple Weighted LSTM (Token + POS Tag)
```
python main.py --exp multi_weighted --test --rnn_num 2 --cuda
```

5. Multiple Unweighted LSTM (Token + POS Tag)
```
python main.py --exp multi_unweighted --test --rnn_num 2 --simple --cuda
```

## Generate
```
python generate.py --exp lm --rnn_num 1 --lm --outf ../results/lm_generated.txt --cuda
```

