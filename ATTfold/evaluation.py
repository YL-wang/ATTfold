from ATTfold.common.utils import *
import csv


def all_train(train_generator, ATTfold, device):
    ATTfold.eval()
    result = list()
    result_shift = list()

    for contacts, seq_embeddings, seq_lens in train_generator:
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)

        PE_batch = get_pe(seq_lens, contacts.shape[-1]).float().to(device)
        with torch.no_grad():
            pred_contacts, a_pred = ATTfold(PE_batch,
                                                  seq_embedding_batch)
        # the learning pp result
        final_pred = (a_pred.cpu() > 0.5).float()
        result_tmp = list(map(lambda i: evaluate_exact(final_pred.cpu()[i],
                                                       contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result += result_tmp

    exact_p,exact_r,exact_f1 = zip(*result)

    exact_p = np.nan_to_num(np.array(exact_p))
    exact_r = np.nan_to_num(np.array(exact_r))
    exact_f1 = np.nan_to_num(np.array(exact_f1))

    precision = np.average(exact_p)
    recall = np.average(exact_r)
    f1 = np.average(exact_f1)

    print('Average train F1 score : ', f1)
    print('Average train precision : ', precision)
    print('Average train recall : ', recall)


def all_val(val_generator, ATTfold, device):
    ATTfold.eval()
    result = list()
    result_shift = list()

    for contacts, seq_embeddings,  seq_lens in val_generator:

        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)

        PE_batch = get_pe(seq_lens, contacts.shape[-1]).float().to(device)
        with torch.no_grad():
            pred_contacts, a_pred = ATTfold(PE_batch,
                                        seq_embedding_batch)
        # the learning pp result
        final_pred = (a_pred.cpu()>0.5).float()
        result_tmp = list(map(lambda i: evaluate_exact(final_pred.cpu()[i],
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result += result_tmp

    exact_p,exact_r,exact_f1 = zip(*result)
    exact_p = np.nan_to_num(np.array(exact_p))
    exact_r = np.nan_to_num(np.array(exact_r))
    exact_f1 = np.nan_to_num(np.array(exact_f1))

    precision = np.average(exact_p)
    recall = np.average(exact_r)
    f1 = np.average(exact_f1)

    print('Average val F1 score : ', f1)
    print('Average val precision : ', precision)
    print('Average val recall : ', recall)

    val_f1 = f1
    return val_f1

def all_test(ATTfold, test_generator, test_data, device):
    ATTfold.eval()
    result = list()

    for contacts, seq_embeddings, seq_lens in test_generator:

        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)

        PE_batch = get_pe(seq_lens, contacts.shape[-1]).float().to(device)
        with torch.no_grad():
            pred_contacts, a_pred = ATTfold(PE_batch,
                                        seq_embedding_batch)

        # the learning pp result
        final_pred = (a_pred.cpu() > 0.5).float()
        result_tmp = list(map(lambda i: evaluate_exact(final_pred.cpu()[i],
                                                       contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result += result_tmp

    exact_p, exact_r, exact_f1 = zip(*result)

    exact_p = np.nan_to_num(np.array(exact_p))
    exact_r = np.nan_to_num(np.array(exact_r))
    exact_f1 = np.nan_to_num(np.array(exact_f1))

    print('Average testing F1 score : ', np.average(exact_f1))
    print('Average testing precision : ', np.average(exact_p))
    print('Average testing recall : ', np.average(exact_r))

    result_df = pd.DataFrame()
    result_df['name'] = [a.name for a in test_data.data]
    result_df['type'] = list(map(lambda x: x.split('/')[3], [a.name for a in test_data.data]))
    result_df['precision'] = exact_p
    result_df['recall'] = exact_r
    result_df['f1'] = exact_f1
    for rna_type in result_df['type'].unique():
        print(rna_type)
        df_temp = result_df[result_df.type == rna_type]
        to_output = list(map(str,
                             list(df_temp[['f1', 'precision', 'recall'
                                           ]].mean().values.round(3))))
        print(to_output)

def save_val(ATTfold, test_generator, test_data, device):
    ATTfold.eval()
    result = list()

    for contacts, seq_embeddings, seq_lens in test_generator:

        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)

        PE_batch = get_pe(seq_lens, contacts.shape[-1]).float().to(device)
        with torch.no_grad():
            pred_contacts, a_pred = ATTfold(PE_batch,
                                        seq_embedding_batch)


        final_pred = (a_pred.cpu() > 0.5).float()
        result_tmp = list(map(lambda i: evaluate_exact(final_pred.cpu()[i],
                                                       contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result += result_tmp

    exact_p, exact_r, exact_f1 = zip(*result)

    exact_p = np.nan_to_num(np.array(exact_p))
    exact_r = np.nan_to_num(np.array(exact_r))
    exact_f1 = np.nan_to_num(np.array(exact_f1))

    precision = np.average(exact_p)
    recall = np.average(exact_r)
    f1 = np.average(exact_f1)

    print('Average test F1 score : ', f1)
    print('Average test precision : ', precision)
    print('Average test recall : ', recall)

    result_df = pd.DataFrame()
    result_df['name'] = [a.name for a in test_data.data]
    result_df['type'] = list(map(lambda x: x.split('/')[3], [a.name for a in test_data.data]))
    result_df['precision'] = exact_p
    result_df['recall'] = exact_r
    result_df['f1'] = exact_f1

    with open(r"../experiment_rnastralign/map/save_val.txt", "a") as f:
        f.write("\n-------------------------------------Line-----------------------------------------\n")
        f.write('\n' + 'Average test F1 score : '+ str(f1))
        f.write('\n' + 'Average test precision : '+ str(precision))
        f.write('\n' + 'Average test recall : '+ str(recall))

        for rna_type in result_df['type'].unique():
            print(rna_type)
            df_temp = result_df[result_df.type == rna_type]
            to_output = list(map(str,
                                 list(df_temp[['f1', 'precision', 'recall'
                                               ]].mean().values.round(3))))
            f.write('\n' + rna_type)
            f.write('\n' + str(to_output))


            print(to_output)
