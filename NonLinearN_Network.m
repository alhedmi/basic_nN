clc; clear; close all;


q1_test = zscore(myRandVector(30,-5,5));
q2_test = zscore(myRandVector(30,-5,5));
q1_sorted = sort(q1_test);
q2_sorted = sort(q2_test);

%number of epoch
%numberOfEpoch = 10^6;
numberOfEpoch = 100000;
%number of data samples (vector of numberOfSamples rows and 1 column)
numberOfSamples = 100;

range_container=zeros(1,numberOfEpoch);
training_container = zeros(1,numberOfEpoch);
learning_rate_container = zeros(1,numberOfEpoch);
mean_learning_rate_container = zeros(1,numberOfEpoch);
mean_Grad_container = zeros(1,numberOfEpoch);



%our desired learning velocity
Initial_learningRate = 10^-3; %-3
decay = 0.0000001;
init_decay = 10^-9;
%create our input values randomly  
q1 = zscore(myRandVector(numberOfSamples,-15,15))';
q2 = zscore(myRandVector(numberOfSamples,-15,15))';
%create the grountruth data (measured data with the targeted/searched dynamics)
x1LabelVector = (q1 + (q2.*q2) -1)';
x2LabelVector = (q1 - q2 +1)';
%create our vector of labels (measured data)
xLabelVector = [x1LabelVector x2LabelVector];

%initialization of our vector of weights
w = zscore(myRandVector(29,-0.05,0.05));
%our vector of input data (with numberOfSamples rows and two columnn).
uVector = [q1' q2'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%NEW%%%%%%%%%%%%%%%%%%%%
alpha = 0.0005; % = r
mBeta_1 = 0.95;    % decay rate w.r.t first moment
mBeta_2 = 0.9995;  % decay rate w.r.t second moment 
epsilon = 1e-8; % against zero division
mk = zeros(size(w)); % first moment estimate
v_k = zeros(size(w)); % second moment estimate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up figure for dynamic visualization
fig = figure;
ax1 = subplot(3, 3, [1, 2, 3]); % For network visualization
ax2 = subplot(3, 3, 4); % For learning rate plot
ax3 = subplot(3, 3, 5); % For training error plot
ax4 = subplot(3, 3, 6); % For Gradiant plot
ax5 = subplot(3, 3, [7,8]); % For X1 Function plot
ax6 = subplot(3, 3, 9); % For X2 Function plot


% we go through the entire dataset numberOfEpoch times in the following loop
for i = 1:numberOfEpoch
    %The three key steps of the training objective

    %Step (1) 
    %Obtain the output of the network for all samples and the current
    %vector of weights being optimized
    [x1Vector, x2Vector] = forwardVector (uVector, w);
    
    %Step (2) 
    % Compute the gradient vector associated with the
    % current input (the samples) of the network and associated 
    % labels (measured data) as well as the current weight
    % vector (the w vector). Observe that the returned Loss is w.r.t all
    % the samples
    [grad, returnedLoss] = calcGradientOfLossFunction(uVector, xLabelVector, w);
    
    %#############################################################################################
    % ##### Uncomment this if you want to try out (self learning) Adaptive Hyperparameters for the ADAM method...
    %         ...THOUGH beware that you'll have to change some things.######
    
    % Adaptive Hyperparameters:    

    % % Update alpha (stepsize) based on loss trends
    % if i > 1
    % delta_loss = training_container(i-1) - lossValue;
    % 
    %     if delta_loss > 0
    %         alpha = alpha * 1.01;
    %     else
    %         alpha = alpha * 0.9;
    %     end
    % 
    %     alpha = alpha / (1 + (0.1 ^ (0.01*i)));
    %     alpha = max(min(alpha, 0.01), 1e-5);
    % end
    % 
    % gradient_variance = var(grad);
    % mBeta_1 = max(0.85, min(0.95, 0.9 + 1 * gradient_variance)); 
    % mBeta_2 = max(0.97, min(0.999, 0.99 + 1 * gradient_variance)); 
    % ####################################################################################################

    %Step (3) 
    % Optimize the weights of the neural network using the previously calculated gradient vector.
    % Once this has been done, jump to Step (1) and again and gain to steer
    % the weights (w) toward a direction for which the neural network is going
    % to capture the underlying model entailed in the data.

    % grad = grad / (norm(grad) + 1e-8); % Normalize to avoid tiny updates


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ADAM %%%%%%%%%%%%%%%%%%%%%%%%%%%
        % w.r.t first
    mk = mBeta_1 * mk + (1 - mBeta_1)*grad;

    % w.r.t second
    v_k = mBeta_2 * v_k + (1 - mBeta_2)*(grad.^2);

    % corrective adaptation
    mHat_k = mk/(1 - mBeta_1^i);
    vHat_k = v_k/(1 - mBeta_2^i);

    % adaptive learning rate
    rk = min(alpha./ (sqrt(vHat_k + epsilon)),0.01);

    % weight update
    w = w - (rk.* mHat_k); %instead of wk = wk - r*grad;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

    % %----------------------Dynamic Learnning Rate Decay ------------------------------------------
    % modifier = 1;
    % decay_dyn = init_decay/(1 - (returnedLoss*modifier));
    % learningRate = Initial_learningRate / (1 - (decay_dyn^2));
    % w = w - learningRate.*grad;
    % % -----------------------------------------------------------------
    
    % The following is just for visulaization purposes, making sure that the
    % user is aware of the current state of the parameter (i.e., weight) optimization
    disp(['         Max rk: ', num2str(max(rk)), ', Min rk: ', num2str(min(rk)), ', Mean rk: ', num2str(mean(rk))]);
    
    num_capped = sum(rk == 0.01);
    disp(['         Number of capped values at 0.01: ', num2str(num_capped)]);

        
    % process
    lossValue = loss (x1LabelVector, x2LabelVector, x1Vector, x2Vector);
    fprintf("step: %i ===== loss:  %10.10f \n\n",i, lossValue);

     % Update training error and LearningRate plot
    range_container(i) =  i;
    training_container(i) = lossValue;
    
    
    % learning_rate_container(i) = rk;
    mean_learning_rate_container(i) = mean(rk);
    mean_Grad_container(i) = mean(grad);
    
    % Update plots every 100 iterations
    if  mod(i, 1000) == 0
        plot_dynamic_network2(ax1, [2, 3,3, 2], w);
        
        dis_name = ['Loss: ', num2str(training_container(i))];
        plot(ax3, range_container(1:i), training_container(1:i), 'b-', "LineWidth", 2, 'DisplayName', dis_name);
        legend(ax3, 'Location', 'north');

        xlabel(ax3, 'Iteration');
        ylabel(ax3, 'Training Error');
        title(ax3, 'Training Error Over Time');
        grid(ax3, 'on');
        drawnow;

        % Update learning rate plot
        hold(ax2, 'on');

        % plot(ax2, range_container, learning_rate_container, 'r-', "LineWidth", 2);
        xlabel(ax2, 'Iteration');
        ylabel(ax2, 'Learning Rate');
        title(ax2, 'Learning Rate Over Time');

        grid(ax2, 'on');

        % plot(ax2, range_container(1:i), learning_rate_container(1:i), 'r-', "LineWidth", 1);
        val = num2str(mean_learning_rate_container(i));
        stri = ['Average Learning Rate: ', val];
        disp(stri);
        mean_plot = plot(ax2, range_container(1:i), mean_learning_rate_container(1:i), 'c-', ...
            "LineWidth", 3, 'DisplayName', stri);
        legend(ax2, mean_plot, 'Location','northwest');

        ylim(ax2, [0, 1.2 * max(mean_learning_rate_container)]);

        drawnow;
        hold(ax2, 'off');
        val1 = num2str(mean_Grad_container(i));
        stri1 = ['Average Gradiant: ', val1];
         Grad_plot = plot(ax4, range_container(1:i), mean_Grad_container(1:i), 'g-', ...
            "LineWidth", 3, 'DisplayName', stri1);
        legend(ax4, Grad_plot, 'Location','southeast');

        ylim(ax4, [min(mean_Grad_container)*1.2, 1.2 * max(mean_Grad_container)]);
        xlabel(ax4, 'Iteration');
        ylabel(ax4, 'Gradiant');
        title(ax4, 'Average Gradiant Over Time');
        grid(ax4, 'on');
        
        drawnow;
        
        
        
        soll = [(q1_test+(q2_test.*q2_test)-1),(q1_test-q2_test+1)];
        
        for j = 1:length(q1_test)
            [x1_plot(j), x2_plot(j)]=forward ([q1_test(j),q2_test(j)], w);
        end
        ist = [x1_plot',x2_plot'];
        t = linspace(0,10,length(soll(:,1)));
        Func_plot = plot(ax5,q1_sorted,soll(:,1));
        hold(ax5,'on');
        Func_plot_2 = plot(ax5,q1_sorted,ist(:,1));
        grid(ax5,"on");
        legend(ax5,'X1 Soll','X1 Ist');
        hold(ax5,'off');
        xlabel(ax5, 'Input');
        ylabel(ax5, 'X1 Output');
        title(ax5, 'Expected Function Vs Predicted Function');
        drawnow;

        Func_plot3 = plot(ax6,q1_sorted,soll(:,2));
        hold(ax6,'on');
        Func_plot_4 = plot(ax6,q1_sorted,ist(:,2));
        grid(ax6,"on");
        legend(ax6,'X2 Soll','X2 Ist');
        hold(ax6,'off');
        xlabel(ax6, 'Input');
        ylabel(ax6, 'X2 Output');
        title(ax6, 'Expected X2 Function Vs Predicted Function');
        drawnow;
    end
end

%%
%quick test
q1 = 0.75;
q2 = -0.25;
%we expect x1 = q1 + q2 - 1 = -0.5 and x2 = q1 - q2 + 1 = 2
disp("q1: "+num2str(q1));
disp("q2: "+num2str(q2));
[x1, x2]=forward ([q1,q2], w);
fprintf("Quick Verification: x1= %10.10f ===== x2=  %10.10f \n",x1, x2);
fprintf("Expected: x1= %10.10f ===== x2=  %10.10f \n\n",(q1+(q2*q2)-1) , (q1-q2+1));

%%
% This function is going to compute the loss by using input, labels and
% frozen weights (that is, it does not modify the w)
function [lossResult] = lossFromForwardVector (uVector, xLabels, w)
    [x1Vector, x2Vector] = forwardVector (uVector, w);
    R_Term =  0.01 * sum(w.^2);
    lossResult = loss (xLabels(:,1), xLabels(:,2), x1Vector, x2Vector) + R_Term; 

end

%This function computes the output of our neural network for given inputs (i.e., vector of single inputs),
%and vector of weights (i.e., w) and returns the vector of corresponding outputs
function [x1Vector, x2Vector] = forwardVector (uVector, w)
    n = size(uVector, 1);
    x1Vector = zeros(n,1);
    x2Vector = zeros(n,1);
    for i = 1:n
         [x1,x2] = forward(uVector(i,:), w);
         x1Vector(i) = x1;
         x2Vector(i) = x2;
    end
end

%This function computes the output of our neural network for given inputs (i.e., single input and output values),
%and vector of weights (i.e., w) and returns the vector of corresponding single
%outputs (i.e., two sclars)

function [x1, x2] = forward (u, w)
q1 = u(1);
q2 = u(2);

a = 0.7;
b = 0.3;
% Ensure 'w' has enough elements
assert(length(w) == 29, 'The weight vector w must have 42 elements.');


% Weights and biases for input to first hidden layer (2 → 3)
w11_0 = w(1);  w12_0 = w(2);  
w21_0 = w(3);  w22_0 = w(4);  
w31_0 = w(5);  w32_0 = w(6);  
w1b_0 = w(7);  
w2b_0 = w(8);  
w3b_0 = w(9);  

% Weights and biases for first hidden to second hidden layer (3 → 3)
w11_1 = w(10); w12_1 = w(11); w13_1 = w(12);
w21_1 = w(13); w22_1 = w(14); w23_1 = w(15);
w31_1 = w(16); w32_1 = w(17); w33_1 = w(18);
w1b_1 = w(19);  
w2b_1 = w(20);  
w3b_1 = w(21);  

% Weights and biases for second hidden to output layer (3 → 2)
w11_2 = w(22); w12_2 = w(23);
w21_2 = w(24); w22_2 = w(25);
w31_2 = w(26); w32_2 = w(27);
w1b_2 = w(28);  
w2b_2 = w(29);  



% Forward pass: Input to First Hidden Layer
x1_1 = myActivation_2(w11_0*q1 + w21_0*q2 + w1b_0, a, b);
x2_1 = myActivation_2(w12_0*q1 + w22_0*q2 + w2b_0, a, b);
x3_1 = myActivation_2(w31_0*q1 + w32_0*q2 + w3b_0, a, b);

% Forward pass: Hidden Layer 1 to Hidden Layer 2
x1_2 = myActivation_1(w11_1*x1_1 + w21_1*x2_1 + w31_1*x3_1 + w1b_1, a, b);
x2_2 = myActivation_1(w12_1*x1_1 + w22_1*x2_1 + w32_1*x3_1 + w2b_1, a, b);
x3_2 = myActivation_1(w13_1*x1_1 + w23_1*x2_1 + w33_1*x3_1 + w3b_1, a, b);

% Forward pass: Second Hidden Layer to Output Layer
x1 = myActivation(w11_2*x1_2 + w21_2*x2_2 + w31_2*x3_2 + w1b_2, a, b);
x2 = myActivation(w12_2*x1_2 + w22_2*x2_2 + w32_2*x3_2 + w2b_2, a, b);


end

%The activation functions, you might want to extend some of the functions or add new ones to take
%additional activitation functions under account


function [outputParameter]=myActivation(inputParameter, a, b)

    if(inputParameter > 0)
        outputParameter = a*inputParameter;
    else
        outputParameter = b*inputParameter; 
    end
end

function [outputParameter]=myActivation_1(inputParameter, ~, ~)
    alpha = 1;
    if(inputParameter > 0)
        outputParameter = inputParameter;
    else
        outputParameter = alpha*(exp(inputParameter)-1); 
    end


end

function [outputParameter]=myActivation_2(inputParameter, a, b)
    if(inputParameter > 0)
        outputParameter = a*(inputParameter*inputParameter);
    else
        outputParameter = -b*(inputParameter*inputParameter);
    end
end




%This function compute the loss and the sum of the single losses over all
%the data samples
function [rloss] = loss (x1LabelVector, x2LabelVector, X1Vector, X2Vector)
%x1LabelVector = grounturth X1
%x2LabelVector = grounturth X2
%X1Vector = x1 output of the network
%X2Vector = x2 output of the network

    rloss = sum( (x1LabelVector - X1Vector).^2 + (x2LabelVector - X2Vector).^2 );

end

%This function compute the gradient of the loss w.r.t to the weights (i.e., the w)
function [gradient, loss] = calcGradientOfLossFunction(inputTrainingVector, outputLabelVector, w)
    % Calculate the gradient of the loss function
    % with respect to the parameters (trained Weights)
    
    % Define a small step size 
    espsilonValue = 1e-3; %1e-6;
    
    trainedWeights = w;

    % Initialize the gradient vector
    gradient = zeros(size(trainedWeights)); 
    
    % Compute the contribution to the gradient of the loss for each weight
    for i = 1:length(trainedWeights)
        % Perturb the current weight using the espsilonValue to the right
        trainedWeightsPerturbedRight = trainedWeights;
        trainedWeightsPerturbedRight(i) = trainedWeightsPerturbedRight(i) + espsilonValue;
        
        % Calculate loss values for wright-perturbed parameter
        [costFunctionPertubedRight]= lossFromForwardVector(inputTrainingVector,outputLabelVector, trainedWeightsPerturbedRight);
        
        % Perturb the parameter value by -espsilonValue (ie, to the left)
        trainedWeightsPerturbedLeft = trainedWeights;
        trainedWeightsPerturbedLeft(i) = trainedWeightsPerturbedLeft(i) - espsilonValue;
        
        % Calculate loss values for perturbed parameter
        [costFunctionPertubedLeft] = lossFromForwardVector(inputTrainingVector, outputLabelVector, trainedWeightsPerturbedLeft);
        
        % Approximate the gradient contribution 
        gradient(i) = (costFunctionPertubedRight - costFunctionPertubedLeft)... 
        / (2 * espsilonValue);
        %pick up the loss
        if(i == length(trainedWeights))
            loss = costFunctionPertubedLeft;
        end
    end
end

