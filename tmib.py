import pandas as pd
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score

def read_train_data(path):
    df_train = pd.read_csv(path)
    df_train = df_train[~df_train.label.isna()]
    return df_train

def read_yok(path):
    df_turkish = pd.read_excel(path)
    df_turkish.loc[df_turkish["labels"] == "NLP", "labels"] = 1
    df_turkish.loc[df_turkish["labels"] == "NOT_NLP", "labels"] = 0

    df_en = pd.DataFrame(columns = ["text", "labels"])
    df_en["text"] = df_turkish["title2"]
    df_en["labels"] = df_turkish["labels"]

    df_tr = pd.DataFrame(columns = ["text", "labels"])
    df_tr["text"] = df_turkish["title1"]
    df_tr["labels"] = df_turkish["labels"]

    return df_en, df_tr


def train(df_train, yok_en, yok_tr):
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    model_args = ClassificationArgs(num_train_epochs=1, train_batch_size=32)
    print("Downloading the model")
    model = ClassificationModel("bert", "bert-base-multilingual-cased", args=model_args, use_cuda=True)
    model.args.use_multiprocessing = False
    model.args.use_multiprocessing_for_evaluation = False
    model.args.multiprocessing_chunksize = 1
    model.args.dataloader_num_workers = 1
    print("Model is downloaded")
    print("Starting the training")
    model.train_model(df_train)
    print("Training is finished")


    return model

def tmib_weighting(model_outputs_en, model_outputs_tr):
    mean_list_w = []
    for idx, item in enumerate(model_outputs_en):
        mean = [(0.44*g + 0.56*h) / 2 for g, h in zip(item, model_outputs_tr[idx])]
        mean_list_w.append(mean)

    return mean_list_w

def get_results(labels, model_outputs_binary):
    accuracy = accuracy_score(list(yok_tr.labels), model_outputs_binary)
    precision = precision_score(list(yok_tr.labels), model_outputs_binary)
    recall = recall_score(list(yok_tr.labels), model_outputs_binary)
    f1 = f1_score(list(yok_tr.labels), model_outputs_binary)
    print("Precision is ", precision)
    print("Recall is ", recall)
    print("F1 is ", f1)
    print("Accuracy is ", accuracy)

if __name__ == "__main__":
    train_data = read_train_data("20220108CLtrain.csv")
    yok_en, yok_tr = read_yok("yok_tr_200_v1_with_labels.xlsx")
    model = train(train_data, yok_en, yok_tr)
    #print(result_en) 
    #print(result_tr)
    #print(model_outputs_en)
    #print(model_outputs_tr)
    
    # Evaluation
    result_en, model_outputs_en, wrong_predictions_en = model.eval_model(yok_en)
    result_tr, model_outputs_tr, wrong_predictions_tr = model.eval_model(yok_tr)

    # Binary Outputs
    model_outputs_en_binary = [0 if item[0] > item[1] else 1 for item in model_outputs_en]
    model_outputs_tr_binary = [0 if item[0] > item[1] else 1 for item in model_outputs_tr]
    
    # We are using yok_tr.labels as labels because the labels if yok_tr and yok_en are same.
    print("These are the Turkish Results :")
    get_results(yok_tr.labels, model_outputs_tr_binary)
    print("These are the English Results :")
    get_results(yok_tr.labels, model_outputs_en_binary)


    # Mean Weighting
    mean_list_w = tmib_weighting(model_outputs_en, model_outputs_tr)
    mean_list_w = [0 if item[0] > item[1] else 1 for item in mean_list_w]

    # The Mean weighting results
    get_results(yok_tr.labels, mean_list_w)

