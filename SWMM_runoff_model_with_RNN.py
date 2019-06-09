import time
import os
import random

import tensorflow as tf
import numpy as np
from runoff_LSTM import runoff_LSTM
from tensorflow.python import debug as tf_debug


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


# W:m; A:m*m; delta_d: m
def get_runoff(W, A, n, S, delta_d):
    runoff = W * np.sqrt(S) * np.power(delta_d, 5/3)
    runoff = np.divide(runoff, A * n)
    return runoff # in: m/s


# all in: m
def R_K_runoff(W, A, n, S, initial_d, delta_t, i, f, Dstore):
    i = i/1000
    f = f/1000
    initial_d = initial_d/1000
    q_0 = get_runoff(W, A, n, S, np.maximum(initial_d - Dstore, 0))
    d_1 = initial_d + 1/5*delta_t*(i-f-q_0)
    q_1 = get_runoff(W, A, n, S, np.maximum(d_1-Dstore, 0))
    d_2 = d_1 + 3/40*delta_t*(i-f-q_0) + 9/40*delta_t*(i-f-q_1)
    q_2 = get_runoff(W,A,n,S, np.maximum(d_2-Dstore, 0))
    d_3 = d_2 + 3/40*delta_t*(i-f-q_0) - 9/10*delta_t*(i-f-q_1) + 6/5*delta_t*(i-f-q_2)
    q_3 = get_runoff(W, A, n, S, np.maximum(d_3-Dstore, 0))
    d_4 = d_3 - 11/54*delta_t*(i-f-q_0) + 5/2*delta_t*(i-f-q_1) - 70/27*delta_t*(i-f-q_2) + 35/27*delta_t*(i-f-q_3)
    q_4 = get_runoff(W, A, n, S, np.maximum(d_4-Dstore, 0))
    d_5 = d_4 + 1631/55296*delta_t*(i-f-q_0) + 175/512*delta_t*(i-f-q_1) + 575/13824*delta_t*(i-f-q_2)+ 44275/110592*delta_t*(i-f-q_3) + 253/4096*(i-f-q_4)
    q_5 = get_runoff(W, A, n, S, np.maximum(d_5-Dstore, 0))
    q = 37/378*q_0 + 250/621*q_2 + 125/594*q_3 + 512/1771*q_5
    return q # in: m/s


# get impervious area without depression's runoff in one time step
# subarea_1_area = area(8)* (1-imperv(2)) * Zero_imperv(3)
# initial_d: mm
def get_runoff_1(inputs, initial_d, delta_t, t_i, i, f):
    t_i = int(t_i)
    subarea_1_area = inputs[:,t_i,8]*(1-inputs[:,t_i,2])*inputs[:,t_i,3] # subarea_1_area: [batch_size]
    subarea_1_slope = inputs[:,t_i,10]
    subarea_1_width = inputs[:,t_i,9]
    subarea_1_N_imperv = inputs[:,t_i,6]
    # i, f shape: [batch_size]
    subarea_1_runoff = R_K_runoff(subarea_1_width, subarea_1_area, subarea_1_N_imperv, subarea_1_slope,initial_d,
                                  delta_t, i, f, 0) # subarea_1_runoff: [batch_size]
    subarea_1_runoff = np.multiply(subarea_1_runoff, subarea_1_area)
    return subarea_1_runoff  # in: m^3/s


# compute impervious area with depression's runoff for depth d
# initial_d in: mm
def get_runoff_2(inputs, initial_d, delta_t, t_i, i, f):
    subarea_2_area = inputs[:, int(t_i), 8] * (1- inputs[:, int(t_i), 2]) * (1 - inputs[:, int(t_i), 3])
    subarea_2_slope = inputs[:, int(t_i), 10]
    subarea_2_width = inputs[:, int(t_i), 9]
    subarea_2_N_imperv = inputs[:, int(t_i), 6]
    subarea_2_Dstore = inputs[:, int(t_i), 4]  # in: m

    subarea_2_runoff = R_K_runoff(subarea_2_width, subarea_2_area, subarea_2_N_imperv, subarea_2_slope, initial_d,
                                  delta_t, i, f, subarea_2_Dstore)
    subarea_2_runoff = np.multiply(subarea_2_runoff,subarea_2_area)
    return subarea_2_runoff


# compute pervious area's runoff
def get_runoff_3(inputs, initial_d, delta_t, t_i, i, f):
    subarea_3_area = inputs[:, int(t_i), 8] * inputs[:, int(t_i), 2]
    subarea_3_slope = inputs[:, int(t_i), 10]
    subarea_3_width = inputs[:, int(t_i), 9]
    subarea_3_N_perv = inputs[:, int(t_i), 7]
    subarea_3_Dstore = inputs[:, int(t_i), 5]

    subarea_3_runoff = R_K_runoff(subarea_3_width, subarea_3_area, subarea_3_N_perv, subarea_3_slope, initial_d,
                                  delta_t, i, f, subarea_3_Dstore)
    subarea_3_runoff = np.multiply(subarea_3_runoff, subarea_3_area)
    return subarea_3_runoff


# compute subarea's runoff for the whole time
def subarea_runoff(inputs, initial_d, initial_t, delta_t, end_t):
    # compute the net inflow volume at every time interval in: mm
    # i is the net inflow rate at every time interval in: mm/s
    # V_prep, V_runon in: [batch_size, max_time]
    V_prep = inputs[:, :, 0]
    V_runon = inputs[:, :, 1]
    i = (V_prep + V_runon)/delta_t*1000

    # f0, fc, beta remain unchanged
    # shape: [batch_size]
    f0 = inputs[:,0, 11]/3600
    fc = inputs[:, 0, 12]/3600
    beta = inputs[:, 0, 13]/3600

    subarea_area = inputs[:, 0, 8]

    # batch_size = tf.shape(inputs)[0]
    # max_time = tf.shape(inputs)[1]
    d_real = np.zeros((10, 1440))  # depth with hydrodynamic equations
    runoff_real = np.zeros((10, 1440))  # runoff with hydrodynamic equations

    initial_p = np.argwhere(i>0)[0,1]*60
    n = (end_t - initial_t)/delta_t + 1
    time_list = np.linspace(initial_t, end_t, n)
    t_i = 0
    # time: means the real initial time at every time interval, i.e. not time index = t_i
    for time in time_list:
        if t_i == 1440:
            continue
        else:
            # compute the infiltration rate(mm/s) at every time interval indexed by t_i
            # for every subcatchment in the batch
            if time < initial_p:
                F = 0
                f = 0
            else:
                t_p = time - initial_p
                F = fc*delta_t - np.divide(f0 - fc, beta)*np.exp(-beta*(t_p + delta_t)) + np.divide(f0 - fc, beta)*np.exp(
                    -beta*t_p)  # same
                f = np.divide(F, delta_t)  # infiltration in: mm
            # i[:t_i] is the net inflow(mm/s)  at every time interval indexed by t_i
            # runoff return by get_runoff_i in: mm^3/s
            # t_i = int(t_i)
            subarea_1_runoff = get_runoff_1(inputs, initial_d, delta_t, t_i, i[:, t_i], f)
            subarea_2_runoff = get_runoff_2(inputs, initial_d, delta_t, t_i, i[:, t_i], f)
            subarea_3_runoff = get_runoff_3(inputs, initial_d, delta_t, t_i, i[:, t_i], f)
            print(t_i)
            print("The net inflow is:")
            print(i[:, t_i]*60)
            runoff = subarea_1_runoff + subarea_2_runoff + subarea_3_runoff
            # use runoff weighted by area in: mm/s
            runoff = np.divide(runoff, subarea_area)
            print("The runoff is")
            print(runoff*delta_t*1000)
            delta_d = i[:, t_i]*delta_t - F - runoff*delta_t*1000
            d_i = np.maximum(initial_d + delta_d, 0)  # d_i: [batch_size]
            print("The infiltration is:")
            print(F)
            print("The depth is:")
            print(d_i)
            # save runoff, depth at t_i
            d_real[:,t_i] = np.reshape(d_i, [10])
            runoff_real[:, t_i] = np.reshape(runoff, [10])
            t_i += 1
            initial_d = d_i

    runoff_real = np.float32(runoff_real)
    d_real = np.float32(d_real)
    # runoff_real = tf.transpose(runoff_real)
    # d_real = tf.transpose(d_real)

    return runoff_real, d_real  # in: mm, mm


def loss(inputs, d_predict, initial_d, initial_t, delta_t, end_t):
    #runoff_real, d_real = tf.py_func(subarea_runoff, [inputs,initial_d,initial_t, delta_t, end_t],
                                     #[tf.float64, tf.float64])
    runoff_real, d_real = tf.py_func(subarea_runoff, [inputs,initial_d,initial_t, delta_t, end_t], [tf.float32, tf.float32])
    # shape:[batch_size, max_time]
    loss_p = 0.5*tf.multiply(d_predict - d_real, d_predict - d_real) # loss: [batch_size, max_time]
    loss_p = tf.reduce_sum(loss_p)
    print("Loss is:")
    print(loss_p)
    return loss_p


# read_data return a list of shape (total_number, max_time, input_depth)
def read_data(filename):
    lines = []
    for i in range(1,711):  # we have 710 subcatchments
        filename_temp = filename + str(i) + ".txt"
        with open(filename_temp, "r") as file:
            lines.append([])
            while True:
                line = file.readline()
                if not line:
                    break
                line = [float(i) for i in line.split()]
                lines[i-1].append(line)
        random.shuffle(lines)
    return lines


def read_batch(stream, batch_size):
    batch = []
    for element in stream:
        batch.append(element)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch


class runoff_model(object):
    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, [10, 1440, 14])
        self.num_units = 128
        self.batch_size = 10
        self.lr = 0.00001
        self.skip_step = 1
        self.num_step = 1440
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        # inputs are: V_prep(0), V_runon(1),
        #               imperv(2), Zero_imperv(3),
        #                Dstore_imperv(4), Dstore_perv(5)
        #                N_imperv(6), N_perv(7)
        #                area(8), width(9), slope(10),
        #                f0(11), fc(12), beta(13)
        #    input_depths = 2+2+2+2+3+3 = 14

    def create_runoff_cell(self, inputs):
        cell = runoff_LSTM(self.num_units)
        # inputs: [batch_size, max_time, input_depth]
        batch = tf.shape(inputs)[0]
        # initialize the hidden state
        zero_states = cell.zero_state(batch, dtype=tf.float32)
        self.in_state = zero_states
        # self.in_state = tuple([tf.placeholder_with_default(state, [None, state.shape[1]]) for state in zero_states])
        length = tf.reduce_sum(tf.reduce_max(tf.sign(inputs), 2), 1)
        self.output, self.out_state = tf.nn.dynamic_rnn(cell, inputs, length, self.in_state)
        # inputs: [batch_size, max_time, ...] = [batch_size, max_time, input_depth]
        # depth: output: [batch_size, max_time, _num_units]
        # out_state: [batch_size, 2*_num_units]

    def create_runoff_model(self):
        self.create_runoff_cell(self.inputs)  # output: [batch_size, max_time, _num_units]
        d_predict = tf.reduce_mean(self.output, axis=2)
        self.mse = loss(self.inputs, d_predict, 0.0, 0, 60, 86400)/10
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.mse, global_step=self.gstep)

    def train(self):
        saver = tf.train.Saver()
        start = time.time()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type="readline")
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
            # get_checkpoint_state gets checkpoint from a directory
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            iteration = self.gstep.eval()
            data_stream = read_data(filename="dataset/Result_3a/Subcatch_S")
            print("Data loaded successfully")
            data = read_batch(data_stream, self.batch_size)
            while True:
                batch = next(data)
                batch = np.array(batch)
                if batch is not None:
                    batch_loss, _ = sess.run([self.mse, self.optimizer],{self.inputs:batch})
                    if (iteration+1) % self.skip_step == 0:
                        print("Iter {}.\n Loss {}. Time {}".format(iteration, batch_loss, time.time()-start))
                        start = time.time()
                        checkpoint_name = 'checkpoints/runoff-model'
                        saver.save(sess, checkpoint_name, iteration)
                    iteration += 1
                else:
                    break


def main():
    model = 'runoff-model'
    safe_mkdir('checkpoints')
    safe_mkdir('checkpoints/' + model)

    hydro_run = runoff_model()
    hydro_run.create_runoff_model()
    hydro_run.train()

if __name__ == '__main__':
    main()
