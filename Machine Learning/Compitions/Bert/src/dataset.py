import config
import torch
import numpy as np
import pandas as pd

class TweetDataset:
    def __init__(self,tweet,sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.max_len = config.MAX_LEN
        self.tokenizer  = config.TOKENIZER
    
    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self,item):
        tweet = " ".join(str(self.tweet[item]).split())
        selected_text = " ".join(str(self.selected_text[item]).split())

        len_sel_text = len(selected_text)
        idx0 = -1
        idx1 = -1
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
            if tweet[ind: ind + len_sel_text] == selected_text:
                idx0 = ind
                idx1 = ind + len_sel_text-1
                break
        
        char_targets = [0] * len(tweet)
        if idx0 != -1 and idx1 != -1:
            for j in range(idx0,idx1+1):
                if tweet[j] != " ":
                    char_targets[j] = 1

        tok_tweet = self.tokenizer.encode(tweet)
        tok_tweet_tokens  = tok_tweet.tokens
        tok_tweet_ids  = tok_tweet.ids
        tok_tweet_offsets  = tok_tweet.offsets[1:-1]

        targets = [0]* (len(tok_tweet_tokens)-2)
        # [0,0,1,1,1,0,0]
        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
            if sum(char_targets[offset1:offset2]) > 0:
                targets[j] =1
        
        # [0,0,0,0,0,0,0]
        targets = [0] + targets + [0] # cls, sep
        targets_start = [0] * len(targets)
        targets_end = [0] * len(targets)

        non_zero = np.nonzero(targets)[0]
        if len(non_zero) > 0:
            targets_start[non_zero[0]] = 1
            targets_start[non_zero[-1]] = 1

        mask = [1] * len(tok_tweet_ids)
        token_type_ids = [0]  * len(tok_tweet_ids)

        padding_len = self.max_len - len(tok_tweet_ids)
        ids = tok_tweet_ids + [0] * padding_len
        mask = mask + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len
        targets = targets + [0] * padding_len
        targets_start = targets_start + [0] * padding_len
        targets_end = targets_end + [0] * padding_len

        # use sentiment 
        sentiment = [1,0,0]
        if self.sentiment[item] == "positive":
            sentiment = [0,0,1]
        
        if self.sentiment[item] == "negative":
            sentiment = [0,1,0]

        return {
            "ids": torch.tensor(ids, dtype = torch.long),
            "mask": torch.tensor(mask, dtype = torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype = torch.long),
            "targets": torch.tensor(targets, dtype = torch.long),
            "targets_start": torch.tensor(targets_start, dtype = torch.long),
            "targets_end": torch.tensor(targets_end, dtype = torch.long),
            "padding_len": torch.tensor(padding_len, dtype = torch.long),
            "tweet_tokens": " ".join(tok_tweet_tokens),
            "orig_tweet": self.tweet[item],
            "sentiment": torch.tensor(sentiment,dtype=torch.long),
            "orig_sentiment": self.sentiment[item],
            "orig_selected": self.selected_text[item]
            
        }

if __name__ == '__main__':
    df = pd.read_csv(config.TRAINING_FILE).dropna().reset_index(drop=True)
    dset = TweetDataset(
        tweet = df.text.values,
        sentiment = df.sentiment.values,
        selected_text = df.selected_text.values
    )
    print (dset[0])

