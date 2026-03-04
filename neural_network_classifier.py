import numpy as np, pandas as pd, copy, matplotlib.pyplot as plt

class NeuralNetworkClassifier:
    def __init__(self, input_size, num_classes, hidden_layers = (64, 32, 16, 8, 4, 2), epochs = 100, alpha = 0.1, l2 = 0.01, clipping_percentile = 90, momentum = 0.9, learning_rate = 0.01, learning_rate_decay_rate = 0.001, ema_smoothing = 2, optimizer = "mbgd"):
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_layers = np.array(hidden_layers)
        self.model_depth = np.concatenate([self.hidden_layers, np.array([num_classes])])
        self.input_weight_size_by_layer = np.concatenate([np.array([self.input_size]), self.hidden_layers])
        self.initialize_parameters()
        self.initialize_parameter_velocities()
        self.epochs = epochs
        self.alpha = alpha
        self.l2 = l2
        self.clipping_percentile = clipping_percentile
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.ema_smoothing = ema_smoothing
        self.optimizer = optimizer
        self.ema_pre_scaled_activation_norm_means = []
        self.ema_pre_scaled_activation_norm_stds = []
        self.batch_step = 1

    def initialize_parameters(self):
        self.weights = [np.array([np.concatenate([[1], [np.random.normal(0, np.sqrt(2.0 / self.input_weight_size_by_layer[i])) for k in range(self.input_weight_size_by_layer[i])]]) for j in range(self.model_depth[i])]) for i in range(len(self.model_depth))]
        self.norm_movers = [np.array([np.random.normal(0, np.sqrt(2.0 / self.input_weight_size_by_layer[i])), 0]) for i in range(len(self.model_depth) - 1)]

    def initialize_parameter_velocities(self):
        self.weight_velocities = [np.array([np.concatenate([[0], [0 for k in range(self.input_weight_size_by_layer[i])]]) for j in range(self.model_depth[i])]) for i in range(len(self.model_depth))]
        self.norm_mover_velocities = [np.array([0, 0]) for i in range(len(self.model_depth) - 1)]

    def adjust_learning_rate(self, i):
        self.learning_rate = self.learning_rate * np.exp(-1 * self.learning_rate_decay_rate * (i + 1))

    def run_forward_pass_batch_training(self, batch):
        weighted_sums_across_layers = []
        pre_scaled_activations_across_layers = []
        pre_scaled_activation_norm_means = []
        pre_scaled_activation_norm_stds = []
        scaled_activations_across_layers = []
        activation_inputs_across_layers = []
        transformed_batch = np.concatenate([np.array([[1 for i in range(len(batch.T[0]))]]), batch.T])
        for i in range(len(self.weights)):
            if i == 0:
                weighted_sums = np.dot(self.weights[i], transformed_batch)
            else:
                weighted_sums = np.dot(self.weights[i], activation_inputs_across_layers[i - 1])
            if i != len(self.weights) - 1:
                pre_scaled_activations = np.where(weighted_sums < 0, self.alpha * weighted_sums, weighted_sums)
                if self.optimizer != "sgd":
                    pre_scaled_activation_batch_layer_means = np.mean(pre_scaled_activations, axis = 1).reshape(-1, 1)
                    pre_scaled_activation_batch_layer_stds = np.std(pre_scaled_activations, axis = 1).reshape(-1, 1)
                else:
                    pre_scaled_activation_batch_layer_means = np.mean(pre_scaled_activations, axis = 0)
                    pre_scaled_activation_batch_layer_stds = np.std(pre_scaled_activations, axis = 0)
                pre_scaled_activation_norm_means.append(pre_scaled_activation_batch_layer_means)
                pre_scaled_activation_norm_stds.append(pre_scaled_activation_batch_layer_stds)
                scaled_activations = (((pre_scaled_activations - pre_scaled_activation_batch_layer_means) / pre_scaled_activation_batch_layer_stds) * self.norm_movers[i][0]) + self.norm_movers[i][1]
            else:
                pre_scaled_activations = np.exp(weighted_sums) / np.sum(np.exp(weighted_sums), axis = 0)
                scaled_activations  = pre_scaled_activations * 1
            weighted_sums_across_layers.append(weighted_sums)
            pre_scaled_activations_across_layers.append(pre_scaled_activations)
            scaled_activations_across_layers.append(scaled_activations)
            if i != len(self.weights) - 1:
                activation_inputs = np.concatenate([np.array([[1 for i in range(len(scaled_activations[0]))]]), scaled_activations])
                activation_inputs_across_layers.append(activation_inputs)
        outputs = self.classes[np.argmax(scaled_activations_across_layers[-1], axis = 0)]
        return transformed_batch, weighted_sums_across_layers, pre_scaled_activations_across_layers, pre_scaled_activation_norm_means, pre_scaled_activation_norm_stds, scaled_activations_across_layers, activation_inputs_across_layers, outputs

    def adjust_batch_emas(self, pre_scaled_activation_norm_means, pre_scaled_activation_norm_stds):
        if self.ema_pre_scaled_activation_norm_means == []:
            self.ema_pre_scaled_activation_norm_means = [np.full(pre_scaled_activation_norm_means[i].shape, 0) for i in range(len(pre_scaled_activation_norm_means))]
        if self.ema_pre_scaled_activation_norm_stds == []:
            self.ema_pre_scaled_activation_norm_stds = [np.full(pre_scaled_activation_norm_stds[i].shape, 0) for i in range(len(pre_scaled_activation_norm_stds))]
        for i in range(len(self.ema_pre_scaled_activation_norm_means)):
            self.ema_pre_scaled_activation_norm_means[i] = (pre_scaled_activation_norm_means[i] * (self.ema_smoothing / (1 + self.batch_step))) - (self.ema_pre_scaled_activation_norm_means[i] * (1 - (self.ema_smoothing / (1 + self.batch_step))))
            self.ema_pre_scaled_activation_norm_stds[i] = (pre_scaled_activation_norm_stds[i] * (self.ema_smoothing / (1 + self.batch_step))) - (self.ema_pre_scaled_activation_norm_stds[i] * (1 - (self.ema_smoothing / (1 + self.batch_step))))

    def compute_gradients(self, transformed_batch, weighted_sums_across_layers, pre_scaled_activations_across_layers, pre_scaled_activation_norm_means, pre_scaled_activation_norm_stds, scaled_activations_across_layers, activation_inputs_across_layers, outputs, batch_target_training_data):
        norm_scalers = [mover[0] for mover in self.norm_movers]
        relu_applied_weighted_sums_scaled_activation_derivative_with_respect_to_pre_scaled_activation = [np.full(pre_scaled_activation_norm_stds[i].shape, norm_scalers[i] / pre_scaled_activation_norm_stds[i]) for i in range(len(pre_scaled_activation_norm_stds))]
        relu_applied_weighted_sums_pre_scaled_activation_derivative_with_respect_to_weighted_sum = [np.where(weighted_sums_across_layers[:-1][i] < 0, self.alpha, 1) for i in range(len(weighted_sums_across_layers[:-1]))]
        output_layer_scaled_activation_derivative_with_respect_to_weighted_sum = np.array([[scaled_activations_across_layers[-1][i][j] * (1 - scaled_activations_across_layers[-1][i][j]) for j in range(len(scaled_activations_across_layers[-1][i]))] for i in range(len(scaled_activations_across_layers[-1]))])
        scaled_activation_derivatives_with_respect_to_weighted_sum = [relu_applied_weighted_sums_scaled_activation_derivative_with_respect_to_pre_scaled_activation[i] * relu_applied_weighted_sums_pre_scaled_activation_derivative_with_respect_to_weighted_sum[i] for i in range(len(relu_applied_weighted_sums_scaled_activation_derivative_with_respect_to_pre_scaled_activation))] + [output_layer_scaled_activation_derivative_with_respect_to_weighted_sum]
        loss_derivatives_with_respect_to_scaled_activation = []
        for i in reversed(range(len(self.model_depth))):
            if i == len(self.model_depth) - 1:
                loss_derivatives_with_respect_to_scaled_activation.append(np.array([[scaled_activations_across_layers[i][j][k] - batch_target_training_data[k][j] for k in range(len(scaled_activations_across_layers[i][j]))] for j in range(len(scaled_activations_across_layers[i]))]))
            else:
                loss_derivatives_with_respect_to_scaled_activation.insert(0, np.array([np.sum(np.array([loss_derivatives_with_respect_to_scaled_activation[0][k] * scaled_activation_derivatives_with_respect_to_weighted_sum[i + 1][k] * self.weights[i + 1][k][1:][j] for k in range(len(self.weights[i + 1]))]), axis = 0) for j in range(len(self.weights[i]))]))
        loss_derivatives_with_respect_to_weighted_sum = [loss_derivatives_with_respect_to_scaled_activation[i] * scaled_activation_derivatives_with_respect_to_weighted_sum[i] for i in range(len(loss_derivatives_with_respect_to_scaled_activation))]
        inputs = [transformed_batch] + [activation_inputs_across_layers[i] for i in range(len(activation_inputs_across_layers))]
        l2_regularizers = [np.array([[np.where(k > 0, 2 * self.l2 * self.weights[i][j][k], 0) for k in range(len(self.weights[i][j]))] for j in range(len(self.weights[i]))]) for i in range(len(loss_derivatives_with_respect_to_weighted_sum))]
        loss_derivatives_with_respect_to_weight = [np.array([[(loss_derivatives_with_respect_to_weighted_sum[i][j][k] * inputs[i].T[k]) + l2_regularizers[i][j] for k in range(len(loss_derivatives_with_respect_to_weighted_sum[i][j]))] for j in range(len(loss_derivatives_with_respect_to_weighted_sum[i]))]) for i in range(len(loss_derivatives_with_respect_to_weighted_sum))]
        final_weight_gradients = [np.array([np.average(loss_derivatives_with_respect_to_weight[i][j], axis = 0) for j in range(len(loss_derivatives_with_respect_to_weight[i]))]) for i in range(len(loss_derivatives_with_respect_to_weight))]
        final_norm_scaler_gradients = np.array([np.sum(np.array([loss_derivatives_with_respect_to_scaled_activation[i] * ((pre_scaled_activations_across_layers[i] - pre_scaled_activation_norm_means[i]) / pre_scaled_activation_norm_stds[i])])) for i in range(len(self.norm_movers))])
        final_norm_shifter_gradients = np.array([np.sum(np.array([loss_derivatives_with_respect_to_scaled_activation[i]])) for i in range(len(self.norm_movers))])
        final_norm_mover_gradients = [np.array([final_norm_scaler_gradients[i], final_norm_shifter_gradients[i]]) for i in range(len(self.norm_movers))]
        return final_weight_gradients, final_norm_mover_gradients

    def set_look_ahead_parameters(self):
        for i in range(len(self.weights)):
            self.weights[i] -= (self.momentum * self.weight_velocities[i])
        for i in range(len(self.norm_movers)):
            self.norm_movers[i] -= (self.momentum * self.norm_mover_velocities[i])

    def set_velocities(self, final_weight_gradients, final_norm_mover_gradients):
        gradients = np.concatenate([np.concatenate([final_weight_gradients[i].flatten() for i in range(len(final_weight_gradients))]), np.concatenate([final_norm_mover_gradients[i].flatten() for i in range(len(final_norm_mover_gradients))])])
        l2_norm = np.linalg.norm(gradients)
        scaled_weight_gradients = [final_weight_gradients[i] * (np.percentile(gradients, self.clipping_percentile) / l2_norm) for i in range(len(final_weight_gradients))] if l2_norm > np.percentile(gradients, self.clipping_percentile) else final_weight_gradients * 1
        scaled_norm_mover_gradients = [final_norm_mover_gradients[i] * (np.percentile(gradients, self.clipping_percentile) / l2_norm) for i in range(len(final_norm_mover_gradients))] if l2_norm > np.percentile(gradients, self.clipping_percentile) else final_norm_mover_gradients * 1
        self.weight_velocities = [(self.momentum * self.weight_velocities[i]) + (self.learning_rate * scaled_weight_gradients[i]) for i in range(len(self.weight_velocities))]
        self.norm_mover_velocities = [(self.momentum * self.norm_mover_velocities[i]) + (self.learning_rate * scaled_norm_mover_gradients[i]) for i in range(len(self.norm_mover_velocities))]
        return scaled_weight_gradients, scaled_norm_mover_gradients

    def run_gradient_descent(self, scaled_weight_gradients, scaled_norm_mover_gradients):
        for i in range(len(self.weights)):
            self.weights[i] -= (self.learning_rate * scaled_weight_gradients[i])
        for i in range(len(self.norm_movers)):
            self.norm_movers[i] -= (self.learning_rate * scaled_norm_mover_gradients[i])

    def run_forward_pass_batch_inference(self, batch):
        weighted_sums_across_layers = []
        pre_scaled_activations_across_layers = []
        scaled_activations_across_layers = []
        activation_inputs_across_layers = []
        transformed_batch = np.concatenate([np.array([[1 for i in range(len(batch.T[0]))]]), batch.T])
        for i in range(len(self.weights)):
            if i == 0:
                weighted_sums = np.dot(self.weights[i], transformed_batch)
            else:
                weighted_sums = np.dot(self.weights[i], activation_inputs_across_layers[i - 1])
            if i != len(self.weights) - 1:
                pre_scaled_activations = np.where(weighted_sums < 0, self.alpha * weighted_sums, weighted_sums)
                scaled_activations = (((pre_scaled_activations - self.ema_pre_scaled_activation_norm_means[i]) / self.ema_pre_scaled_activation_norm_stds[i]) * self.norm_movers[i][0]) + self.norm_movers[i][1]
            else:
                pre_scaled_activations = np.exp(weighted_sums) / np.sum(np.exp(weighted_sums), axis = 0)
                scaled_activations  = pre_scaled_activations * 1
            weighted_sums_across_layers.append(weighted_sums)
            pre_scaled_activations_across_layers.append(pre_scaled_activations)
            scaled_activations_across_layers.append(scaled_activations)
            if i != len(self.weights) - 1:
                activation_inputs = np.concatenate([np.array([[1 for i in range(len(scaled_activations[0]))]]), scaled_activations])
                activation_inputs_across_layers.append(activation_inputs)
        outputs = self.classes[np.argmax(scaled_activations_across_layers[-1], axis = 0)]
        return transformed_batch, weighted_sums_across_layers, pre_scaled_activations_across_layers, scaled_activations_across_layers, activation_inputs_across_layers, outputs

    def run_evaluation(self, input_training_data, input_validation_data, target_training_data, target_validation_data):
        _, _, _, _, _, training_outputs = self.run_forward_pass_batch_inference(input_training_data)
        training_targets = self.classes[np.argmax(target_training_data, axis = 1)]
        training_accuracy = np.sum(np.equal(training_outputs, training_targets)) / len(training_targets)
        _, _, _, _, _, validation_outputs = self.run_forward_pass_batch_inference(input_validation_data)
        validation_targets = self.classes[np.argmax(target_validation_data, axis = 1)]
        validation_accuracy = np.sum(np.equal(validation_outputs, validation_targets)) / len(validation_targets)
        return training_accuracy, validation_accuracy

    def run_batch_training(self, batch_input_training_data, batch_target_training_data):
        self.set_look_ahead_parameters()
        transformed_batch, weighted_sums_across_layers, pre_scaled_activations_across_layers, pre_scaled_activation_norm_means, pre_scaled_activation_norm_stds, scaled_activations_across_layers, activation_inputs_across_layers, outputs = self.run_forward_pass_batch_training(batch_input_training_data)
        self.adjust_batch_emas(pre_scaled_activation_norm_means, pre_scaled_activation_norm_stds)
        self.batch_step += 1
        final_weight_gradients, final_norm_mover_gradients = self.compute_gradients(transformed_batch, weighted_sums_across_layers, pre_scaled_activations_across_layers, pre_scaled_activation_norm_means, pre_scaled_activation_norm_stds, scaled_activations_across_layers, activation_inputs_across_layers, outputs, batch_target_training_data)
        scaled_weight_gradients, scaled_norm_mover_gradients = self.set_velocities(final_weight_gradients, final_norm_mover_gradients)
        self.run_gradient_descent(scaled_weight_gradients, scaled_norm_mover_gradients)

    def train_model(self, input_data, target_data, val_prop = 0.8, mini_batch_size = 32, show = False):
        weights_across_epochs = []
        ema_pre_scaled_activation_norm_means_across_epochs = []
        ema_pre_scaled_activation_norm_stds_across_epochs = []
        training_accuracies_across_epochs = []
        validaton_accuracies_across_epochs = []
        input_training_data = np.array(input_data)[:round(len(input_data) * val_prop)]
        input_validation_data = np.array(input_data)[round(len(input_data) * val_prop):]
        target_encoders = np.array(pd.get_dummies(pd.Series(np.array(target_data)), dtype = float))
        target_training_data = target_encoders[:round(len(target_data) * val_prop)]
        target_validation_data = target_encoders[round(len(target_data) * val_prop):]
        self.classes = np.array(pd.get_dummies(pd.Series(np.array(target_data)), dtype = float).columns)
        if self.optimizer == "mbgd":
            training_data_size = round((len(input_training_data) // mini_batch_size) * mini_batch_size)
            num_batches = round(training_data_size / mini_batch_size)
            input_training_data_across_batches = np.array(np.split(input_training_data[:training_data_size], num_batches))
            target_training_data_across_batches = np.array(np.split(target_training_data[:training_data_size], num_batches))
        for i in range(self.epochs):
            self.adjust_learning_rate(i)
            if self.optimizer == "mbgd":
                random_training_indicies_across_batches = np.random.choice(len(input_training_data_across_batches), len(input_training_data_across_batches), replace = False)
                shuffled_input_training_data_across_batches = input_training_data_across_batches[random_training_indicies_across_batches]
                shuffled_target_training_data_across_batches = target_training_data_across_batches[random_training_indicies_across_batches]
                for j in range(len(shuffled_input_training_data_across_batches)):
                    self.run_batch_training(shuffled_input_training_data_across_batches[j], shuffled_target_training_data_across_batches[j])
            elif self.optimizer == "bgd":
                random_training_indices = np.random.choice(len(input_training_data), len(input_training_data), replace = False)
                shuffled_input_training_data = input_training_data[random_training_indices]
                shuffled_target_training_data = target_training_data[random_training_indices]
                self.run_batch_training(shuffled_input_training_data, shuffled_target_training_data)
            else:
                random_training_index = np.random.choice(len(input_training_data), 1, replace = False)
                random_input_training_data = input_training_data[random_training_index]
                random_target_training_data = target_training_data[random_training_index]
                self.run_batch_training(random_input_training_data, random_target_training_data)
            weights_across_epochs.append(copy.deepcopy(self.weights))
            ema_pre_scaled_activation_norm_means_across_epochs.append(copy.deepcopy(self.ema_pre_scaled_activation_norm_means))
            ema_pre_scaled_activation_norm_stds_across_epochs.append(copy.deepcopy(self.ema_pre_scaled_activation_norm_stds))
            training_accuracy, validation_accuracy = self.run_evaluation(input_training_data, input_validation_data, target_training_data, target_validation_data)
            training_accuracies_across_epochs.append(training_accuracy)
            validaton_accuracies_across_epochs.append(validation_accuracy)
        self.weights = weights_across_epochs[np.argmax(np.array(validaton_accuracies_across_epochs))]
        self.ema_pre_scaled_activation_norm_means = ema_pre_scaled_activation_norm_means_across_epochs[np.argmax(np.array(validaton_accuracies_across_epochs))]
        self.ema_pre_scaled_activation_norm_stds = ema_pre_scaled_activation_norm_stds_across_epochs[np.argmax(np.array(validaton_accuracies_across_epochs))]
        if show:
            plt.plot(validaton_accuracies_across_epochs)
            plt.plot(training_accuracies_across_epochs)
            plt.show()

    def predict(self, batch):
        _, _, _, _, _, outputs = self.run_forward_pass_batch_inference(np.array(batch))
        return outputs
