import pandas as pd

class LabelEncoder:
    def __init__(self):
        self.classes = None
    
    def fit(self, classes):
        unique_classes = list(set(classes))
        unique_classes.sort()
        encoded_label = {}
        for index, label in enumerate(unique_classes):
            if label not in encoded_label:
                encoded_label[label] = index
        self.classes = encoded_label

    def transform(self, row):
        encoded_label = self.classes
        return encoded_label[row] 

if __name__ == '__main__':
    color = ['Red','Red','Blue','Green','Red','White']
    df = pd.DataFrame()
    df['color'] = color
    le = LabelEncoder()
    le.fit(df['color'])
    df['labelled_color'] = df.map(lambda row: le.transform(row))
    print(df)