import numpy as np
import pandas as pd
import collections as col
import scipy
import os
import pickle

from scipy import sparse
from scipy.sparse.linalg import inv
from scipy.sparse import csr, csr_matrix, csc_matrix, hstack, vstack
from time import time
from sklearn.preprocessing import normalize
from sklearn.utils.sparsefuncs import count_nonzero
from tqdm import tqdm, trange
from IPython import embed


class SWALK:
    def __init__(self, reg=10, topK=100, topK_by='G', alpha=0.5, session_weight=-1, train_weight=-1, predict_weight=-1,
                 beta=1, prune=0, steps=1000, sr_weighting='div',
                 model1='SLIT', model2='SLIS',
                 recwalk_method='PR', k_step=-1, p=0.5, recwalk_dense=True, PR_eps=100,
                 last_normalize=False, model_path=None,
                 direction='part', normalize='l1', target_normalize='no', epsilon=10.0,
                 remove_item='no', predict_by='order', session_key='SessionId', item_key='ItemId'):
        self.reg = reg
        self.topK = topK
        self.topK_by = topK_by
        self.normalize = normalize
        self.epsilon = epsilon
        self.remove_item = remove_item
        self.predict_by = predict_by
        self.target_normalize = target_normalize
        self.alpha = alpha
        self.direction = direction  # 'all' or 'part'
        self.train_weight = float(train_weight)
        self.predict_weight = float(predict_weight)
        self.session_weight = session_weight*24*3600

        self.beta = beta
        self.prune = prune
        self.steps = steps
        self.sr_weighting = sr_weighting

        self.model1 = model1
        self.model2 = model2

        self.recwalk_method = recwalk_method
        self.k_step = k_step
        self.p = p
        self.recwalk_dense = recwalk_dense
        self.PR_eps = PR_eps

        self.last_normalize = last_normalize
        self.model_path = model_path

        self.session_key = session_key
        self.item_key = item_key

        # updated while recommending
        self.session = -1
        self.session_items = []

    def fit(self, data, test=None):
        '''
        Trains the predictor.

        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).

        '''
        # make new session ids(1 ~ #sessions)
        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        self.sessionidmap = pd.Series(data=np.arange(self.n_sessions), index=sessionids)
        data = pd.merge(data, pd.DataFrame({self.session_key: sessionids, 'SessionIdx': self.sessionidmap[sessionids].values}), on=self.session_key, how='inner')

        # make new item ids(1 ~ #items)
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        data = pd.merge(data, pd.DataFrame({self.item_key: itemids, 'ItemIdx': self.itemidmap[itemids].values}), on=self.item_key, how='inner')

        #    SessionId          Time  ItemId        Date     Datestamp                             TimeO  ItemSupport  SessionIdx  ItemIdx
        # 0          1  1.462752e+09    9654  2016-05-09  1.462752e+09  2016-05-09 00:01:15.848000+00:00          399           0        0

        def make_SLIST(model):
            input1, target1, row_weight1 = self.make_train_matrix(data, weight_by='SLIS')
            input2, target2, row_weight2 = self.make_train_matrix(data, weight_by='SLIT')

            if model == 'SLIST':
                # alpha * ||X - XB|| + (1-alpha) * ||Y - ZB||
                input1.data = np.sqrt(self.alpha) * input1.data
                target1.data = np.sqrt(self.alpha) * target1.data
                input2.data = np.sqrt(1-self.alpha) * input2.data
                target2.data = np.sqrt(1-self.alpha) * target2.data

                input_matrix = vstack([input1, input2])
                target_matrix = vstack([target1, target2])
                w2 = row_weight1 + row_weight2  # list
                W2 = sparse.diags(w2, dtype=np.float32)
                G = input_matrix.transpose().dot(W2).dot(input_matrix).toarray()
                print("G is made")

                P = np.linalg.inv(G + np.identity(self.n_items, dtype=np.float32) * self.reg)
                print("P is made")
                del G
                enc_w = P @ input_matrix.transpose().dot(W2).dot(target_matrix).toarray()
                return enc_w
            elif model == 'SLIS':
                input_matrix = input1
                target_matrix = target1
                w2 = row_weight1
                W2 = sparse.diags(w2, dtype=np.float32)
                G = input_matrix.transpose().dot(W2).dot(input_matrix).toarray()
                print("G is made")

                P = np.linalg.inv(G + np.identity(self.n_items, dtype=np.float32) * self.reg)
                print("P is made")
                del G
                if self.epsilon < 10.0:
                    gamma = np.zeros(self.n_items)
                    gamma += self.reg
                    mu_nonzero_idx = np.where(1 - np.diag(P)*self.reg >= self.epsilon)
                    gamma[mu_nonzero_idx] = (np.diag(1 - self.epsilon) / np.diag(P))[mu_nonzero_idx]


                    # B = I - P·diagMat(γ)
                    enc_w = np.identity(self.n_items, dtype=np.float32) - P @ np.diag(gamma)
                    print("weight matrix is made")
                else:
                    enc_w = P @ input_matrix.transpose().dot(W2).dot(target_matrix).toarray()
                return enc_w
            elif model == 'SLIT':
                input_matrix = input2
                target_matrix = target2
                w2 = row_weight2
                W2 = sparse.diags(w2, dtype=np.float32)
                G = input_matrix.transpose().dot(W2).dot(input_matrix).toarray()
                print("G is made")

                P = np.linalg.inv(G + np.identity(self.n_items, dtype=np.float32) * self.reg)
                print("P is made")
                del G
                enc_w = P @ input_matrix.transpose().dot(W2).dot(target_matrix).toarray()
                return enc_w

            # P = (X^T * X + λI)^−1 = (G + λI)^−1
            # (A+B)^-1 = A^-1 - A^-1 * B * (A+B)^-1
            # P =  G
            W2 = sparse.diags(w2, dtype=np.float32)
            G = input_matrix.transpose().dot(W2).dot(input_matrix).toarray()
            print("G is made")

            GG = input_matrix.transpose().dot(W2).dot(target_matrix).toarray()
            print("GG is made")

            if self.topK_by == 'G':
                topk_mat = G
            else:
                raise("self.topK_by is not correct")

            enc_w = np.zeros((self.n_items, self.n_items))
            count = 0
            for item in trange(self.n_items):
                topK = min(self.topK, len(np.where(topk_mat[item] > 0)[0]))
                if topK == 0:
                    continue
                if topK < self.topK:
                    count += 1

                item_topK = np.argpartition(-topk_mat[item], topK)[:topK]
                item_in_topK = np.where(item_topK == item)[0]
                if len(item_in_topK) == 0:
                    item_topK[-1] = item
                    item_in_topK = self.topK-1

                # (#items, #items) -> (topK, #items) -> (topK, topK)
                G_for_item = G[np.ix_(item_topK, item_topK)]
                P_for_item = np.linalg.inv(G_for_item + np.identity(topK, dtype=np.float32) * self.reg)
                enc_for_item = P_for_item[item_in_topK] @ GG[np.ix_(item_topK, item_topK)]

                # encoder_for_item = (1, topK)
                enc_w[item, item_topK] = enc_for_item
            return enc_w

        def make_AR():
            ar_matrix = np.zeros((self.n_items, self.n_items))

            for sid, session in data.groupby(['SessionIdx']):
                sessionitems = session['ItemIdx'].tolist()
                for i in sessionitems:
                    for j in sessionitems:
                        ar_matrix[i, j] += 1
            return ar_matrix

        def make_SR():
            sr_matrix = np.zeros((self.n_items, self.n_items))
            sessionlengthmap = data['SessionIdx'].value_counts(sort=False)
            for sid, session in data.groupby(['SessionIdx']):
                slen = sessionlengthmap[sid]
                sessionitems = session.sort_values(['Time'])['ItemIdx'].tolist()  # sorted by time
                for i in range(slen-1):
                    for j in range(i+1, min(slen, (i+1)+self.steps)):
                        if self.sr_weighting == 'div':
                            sr_matrix[sessionitems[i], sessionitems[j]] += 1/(j-i)
                        elif self.sr_weighting == 'quadratic':
                            sr_matrix[sessionitems[i], sessionitems[j]] += 1/((j-i)*(j-i))
                        elif self.sr_weighting == 'one':
                            sr_matrix[sessionitems[i], sessionitems[j]] += 1

            if self.prune > 0:
                self.prune = min(self.prune, self.n_items)
                for i in range(self.n_items):
                    keep = np.argpartition(sr_matrix[i], -self.prune)[-self.prune:]
                    new_sr_i = np.zeros(self.n_items)
                    new_sr_i[keep] = sr_matrix[i, keep]
                    sr_matrix[i] = new_sr_i

            return sr_matrix

        if self.model_path is not None and os.path.exists(self.model_path + f'{self.n_sessions}_{self.n_items}.p'):
            with open(self.model_path + f'{self.n_sessions}_{self.n_items}.p', 'rb') as f:
                self.enc_w = pickle.load(f)
            return

        # model 1
        if self.model1 in ['SLIS', 'SLIT', 'SLIST']:
            model1 = make_SLIST(model=self.model1)
        elif self.model1 == 'AR':
            model1 = make_AR()
        elif self.model1 == 'SR':
            model1 = make_SR()
        elif self.model1 == 'I':
            model1 = np.diag(np.ones(self.n_items))
        else:
            exit()

        # model 2
        if self.model2 in ['SLIS', 'SLIT', 'SLIST']:
            model2 = make_SLIST(model=self.model2)
        elif self.model2 == 'AR':
            model2 = make_AR()
        elif self.model2 == 'SR':
            model2 = make_SR()
        elif self.model2 == 'I':
            model2 = np.diag(np.ones(self.n_items))
        else:
            exit()

        S = model1
        T = model2

        # non-negative matrix
        S[S < 0] = 0
        T[T < 0] = 0

        # For efficiency, we use sparse matrix
        S = csr_matrix(S)
        T = csr_matrix(T)

        # S = Diag(W)^-1 \cdot W
        S = normalize(S, norm='l1', axis=1)
        T = normalize(T, norm='l1', axis=1)

        S = csr_matrix(S)
        T = csr_matrix(T)

        T = self.beta*T + csr_matrix((1-self.beta)*np.diag(np.ones(self.n_items)))

        self.enc_w = self.make_transition_matrix(S, T)

        if self.last_normalize:
            self.enc_w = normalize(self.enc_w, 'l1')

        if self.model_path is not None:
            with open(self.model_path + f'{self.n_sessions}_{self.n_items}.p', 'wb') as f:
                pickle.dump(self.enc_w, f, protocol=4)

    def restore(self, path):
        with open(path, 'rb') as f:
            self.enc_w = pickle.load(f)

    def make_transition_matrix(self, S, T):
        print(f'Sparsity of S: {1 - (S.count_nonzero()/(S.shape[0]*S.shape[1]))}')
        print(f'Sparsity of T: {1 - (T.count_nonzero()/(T.shape[0]*T.shape[1]))}')

        if self.recwalk_method in ['SRW', 'PR']:
            Sk = csr_matrix(np.identity(self.n_items))

            if self.recwalk_method == 'SRW':
                if self.recwalk_dense:
                    S = S.toarray()
                    Sk = Sk.toarray()

                if self.k_step == -1:
                    print(f'Using SRW with k_step = {self.k_step}')
                for _ in tqdm(range(self.k_step), desc=f'{self.recwalk_method}'):
                    if self.recwalk_dense:
                        Sk = S @ Sk
                    else:
                        Sk = S.dot(Sk)
                transition_matrix = Sk
            # 'PR'
            else:
                pk = 1
                # Sigma(0~inf) (1-p)*p^k*S^k == 1-p*1*I + Sigma(1~inf) (1-p)*p^k*S^k
                M = sparse.diags(np.ones(self.n_items), format='csr')
                if self.recwalk_dense:
                    M = M.toarray()
                    S = S.toarray()
                    T = T.toarray()

                k_step = 100 if (self.k_step == -1) else self.k_step
                for _ in tqdm(range(k_step), desc=f'{self.recwalk_method}'):
                    M_last = M.copy()
                    if self.recwalk_dense:
                        M = self.p * (M @ S) + (1-self.p)*T
                    else:
                        M = self.p * (M.dot(S)) + (1-self.p)*T

                    # Check converge, l1 norm
                    err = abs(M_last - M).sum()
                    if err < self.PR_eps:
                        break

                # sparse matrix -> numpy array
                transition_matrix = M
        elif self.recwalk_method in ['SRW_MI', 'PR_MI']:
            # MI = (W/|W|_inf) + Diag(1-(W/|W|_inf))
            # (W/|W|_inf)
            W_normalized = W.copy()
            row_sums = W.sum(axis=1).A1
            row_sums_max = max(row_sums)  # |W|_inf
            # Normalize W by the maximal row sums.
            W_normalized.data /= row_sums_max

            # Original Version------------------
            # Diag(1-(W/|W|_inf))
            # W_norm_diag = sparse.diags(1-(row_sums/row_sums_max), format='csr')
            # MI = W_normalized + W_norm_diag
            # Original Version------------------

            # Exp Version------------------
            W_norm = csr_matrix(1-(row_sums/row_sums_max)).reshape(self.n_items, 1)

            # Add a columns of 1s to W_normalized
            MI = hstack([W_normalized, W_norm])
            # Add a row of zeros to W_normalized
            last_MI_row = csr_matrix(np.zeros((1, self.n_items+1)))
            last_MI_row[0, -1] = 1
            MI = vstack([MI, last_MI_row])
            # Exp Version------------------

            S = MI
            Sk = csr_matrix(np.identity(S.shape[0]))
            if self.recwalk_method == 'SRW_MI':
                if self.recwalk_dense:
                    S = S.toarray()
                    Sk = Sk.toarray()
                for _ in tqdm(range(self.k_step), desc=f'{self.recwalk_method}'):
                    if self.recwalk_dense:
                        Sk = S @ Sk
                    else:
                        Sk = S.dot(Sk)
                transition_matrix = Sk
            elif self.recwalk_method == 'PR_MI':
                # Sigma(0~inf) (1-p)*p^k*S^k == 1-p*1*I + Sigma(1~inf) (1-p)*p^k*S^k
                pk = 1
                PR = sparse.diags(np.ones(S.shape[0]) * (1-self.p), format='csr')
                if self.recwalk_dense:
                    PR = PR.toarray()
                    S = S.toarray()
                    Sk = Sk.toarray()
                for _ in tqdm(range(100), desc=f'{self.recwalk_method}'):
                    PR_last = PR.copy()
                    if self.recwalk_dense:
                        Sk = S @ Sk
                    else:
                        Sk = S.dot(Sk)
                    pk = pk * self.p
                    PR += (1-self.p)*pk*Sk

                    # Check converge, l1 norm
                    err = abs(PR_last - PR).sum()
                    if err < self.PR_eps:
                        break
                # sparse matrix -> numpy array
                transition_matrix = PR
            transition_matrix = transition_matrix[:self.n_items, :self.n_items]
        else:
            exit()

        if not self.recwalk_dense:
            transition_matrix = transition_matrix.toarray()

        return transition_matrix

    def make_train_matrix(self, data, weight_by='SLIT'):
        input_row = []
        target_row = []
        input_col = []
        target_col = []
        input_data = []
        target_data = []

        maxtime = data.Time.max()
        w2 = []
        sessionlengthmap = data['SessionIdx'].value_counts(sort=False)
        rowid = -1

        if weight_by == 'SLIT':
            if os.path.exists(f'./data_ckpt/{self.n_sessions}_{self.n_items}_{self.direction}_SLIT.p'):
                with open(f'./data_ckpt/{self.n_sessions}_{self.n_items}_{self.direction}_SLIT.p', 'rb') as f:
                    input_row, input_col, input_data, target_row, target_col, target_data, w2 = pickle.load(
                        f)
            else:
                for sid, session in tqdm(data.groupby(['SessionIdx']), desc=weight_by):
                    slen = sessionlengthmap[sid]
                    # sessionitems = session['ItemIdx'].tolist() # sorted by itemid
                    sessionitems = session.sort_values(
                        ['Time'])['ItemIdx'].tolist()  # sorted by time
                    if self.remove_item == 'succession':
                        sessionitems = [s for i, s in enumerate(
                            sessionitems) if i != (slen-1) and s != sessionitems[i+1]]
                    elif self.remove_item == 'repeat':
                        # 늦게 등장한 항목을 제거
                        sessionitems = sessionitems[::-1]
                        sessionitems = [s for i, s in enumerate(sessionitems) if i != (
                            slen-1) and s not in sessionitems[i+1:]]
                        sessionitems = sessionitems[::-1]
                    else:
                        pass
                    slen = len(sessionitems)
                    if slen <= 1:
                        continue
                    stime = session['Time'].max()
                    w2 += [stime-maxtime] * (slen-1)
                    for t in range(slen-1):
                        rowid += 1
                        # input matrix
                        if self.direction == 'part':
                            input_row += [rowid] * (t+1)
                            input_col += sessionitems[:t+1]
                            for s in range(t+1):
                                input_data.append(-abs(t-s))
                            target_row += [rowid] * (slen - (t+1))
                            target_col += sessionitems[t+1:]
                            for s in range(t+1, slen):
                                target_data.append(-abs((t+1)-s))
                        elif self.direction == 'all':
                            input_row += [rowid] * slen
                            input_col += sessionitems
                            for s in range(slen):
                                input_data.append(-abs(t-s))
                            target_row += [rowid] * slen
                            target_col += sessionitems
                            for s in range(slen):
                                target_data.append(-abs((t+1)-s))
                        elif self.direction == 'sr':
                            input_row += [rowid]
                            input_col += [sessionitems[t]]
                            input_data.append(0)
                            target_row += [rowid] * (slen - (t+1))
                            target_col += sessionitems[t+1:]
                            for s in range(t+1, slen):
                                target_data.append(-abs((t+1)-s))
                        else:
                            raise ("You have to choose right 'direction'!")
                with open(f'./data_ckpt/{self.n_sessions}_{self.n_items}_{self.direction}_SLIT.p', 'wb') as f:
                    pickle.dump([input_row, input_col, input_data, target_row,
                                 target_col, target_data, w2], f, protocol=4)
            input_data = list(np.exp(np.array(input_data) / self.train_weight))
            target_data = list(
                np.exp(np.array(target_data) / self.train_weight))
        elif weight_by == 'SLIS':
            if os.path.exists(f'./data_ckpt/{self.n_sessions}_{self.n_items}_SLIS.p'):
                with open(f'./data_ckpt/{self.n_sessions}_{self.n_items}_SLIS.p', 'rb') as f:
                    input_row, input_col, input_data, target_row, target_col, target_data, w2 = pickle.load(
                        f)
            else:
                for sid, session in tqdm(data.groupby(['SessionIdx']), desc=weight_by):
                    rowid += 1
                    slen = sessionlengthmap[sid]
                    sessionitems = session['ItemIdx'].tolist()
                    stime = session['Time'].max()
                    # w2.append(np.exp((stime-maxtime)/self.session_weight))
                    w2.append(stime-maxtime)
                    input_row += [rowid] * slen
                    input_col += sessionitems

                target_row = input_row
                target_col = input_col
                input_data = np.ones_like(input_row)
                target_data = np.ones_like(target_row)
                with open(f'./data_ckpt/{self.n_sessions}_{self.n_items}_SLIS.p', 'wb') as f:
                    pickle.dump([input_row, input_col, input_data, target_row,
                                 target_col, target_data, w2], f, protocol=4)
        else:
            raise ("You have to choose right 'weight_by'!")

        # Use train_weight or not
        input_data = input_data if self.train_weight > 0 else list(
            np.ones_like(input_data))
        target_data = target_data if self.train_weight > 0 else list(
            np.ones_like(target_data))

        # Use session_weight or not
        w2 = list(np.exp(np.array(w2) / self.session_weight))
        w2 = w2 if self.session_weight > 0 else list(np.ones_like(w2))

        # Make sparse_matrix
        input_matrix = csr_matrix((input_data, (input_row, input_col)), shape=(
            max(input_row)+1, self.n_items), dtype=np.float32)
        target_matrix = csr_matrix(
            (target_data, (target_row, target_col)), shape=input_matrix.shape, dtype=np.float32)
        print(f"sparse matrix {input_matrix.shape} is made")

        if weight_by == 'SLIT':
            pass
        elif weight_by == 'SLIS':
            # Value of repeated items --> 1
            input_matrix.data = np.ones_like(input_matrix.data)
            target_matrix.data = np.ones_like(target_matrix.data)

        # Normalization
        if self.normalize == 'l1':
            input_matrix = normalize(input_matrix, 'l1')
        elif self.normalize == 'l2':
            input_matrix = normalize(input_matrix, 'l2')
        else:
            pass

        if self.target_normalize == 'l1':
            target_matrix = normalize(target_matrix, 'l1')
        elif self.target_normalize == 'l2':
            target_matrix = normalize(target_matrix, 'l2')
        elif self.target_normalize == 'pop':
            target_matrix = normalize(target_matrix.T, 'l1').T
        else:
            pass

        return input_matrix, target_matrix, w2

    # 필수

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, input_user_id=None, skip=False, type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        '''
        # new session
        if session_id != self.session:
            self.session_items = []
            self.session = session_id
            self.session_times = []

        if type == 'view':
            self.session_items.append(input_item_id)
            self.session_times.append(timestamp)

        # item id transfomration
        session_items_new_id = self.itemidmap[self.session_items].values
        predict_for_item_ids_new_id = self.itemidmap[predict_for_item_ids].values

        if skip:
            return

        W_test = np.ones_like(self.session_items, dtype=np.float32)
        for i in range(len(W_test)):
            W_test[i] = np.exp(-abs(i+1-len(W_test))/self.predict_weight)

        W_test = W_test if self.predict_weight > 0 else np.ones_like(W_test)
        W_test = W_test.reshape(-1, 1)
        # print(W_test)

        if self.predict_by == 'last':
            preds = self.enc_w[session_items_new_id[-1]]
        else:
            # [session_items, num_items]
            preds = self.enc_w[session_items_new_id] * W_test
            # [num_items]
            preds = np.sum(preds, axis=0)

        preds = preds[predict_for_item_ids_new_id]

        series = pd.Series(data=preds, index=predict_for_item_ids)

        series = series / series.max()

        return series

    # 필수
    def clear(self):
        self.enc_w = {}
