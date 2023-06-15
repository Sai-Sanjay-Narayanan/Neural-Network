% defining the layers with no. of units

layers = [6,1];
L = length(layers);

% learning rate
alpha =  0.5*1e-5;
maxiter = 10000;

% cost function -> binary cross entropy loss
J = 0;

% defining activations for each layer
linear = @(x) (x);
dlinear = @(x) (1);

g = cell(1,L);
g{1} = @relu;
%g{2} = @relu;
%g{3} = @relu;
%g{4} = @relu;
g{L} = @sigmoid;

% numerically defining derivatives of activations
tol = 1e-5;
drelu = @(x) (relu(x + tol/2) - relu(x - tol/2))/tol ;
dsigmoid = @(x) (sigmoid(x) .* (1 - sigmoid(x)) );

dg = cell(1,L);
dg{1} = drelu;
%dg{2} = drelu;
%dg{3} = drelu;
%dg{4} = drelu;
dg{L} = dsigmoid;

% data point(s) as column vector(s)

n = length(X(:,1));
m = length(X(1,:));

% defining the weights and biases in each layer

W = cell(1, L);
B = cell(1, L);

% initializing weights and biases randomly

a = -0.1;
b = 0.2;

W{1} = a + b*rand([layers(1), n]);
B{1} = a + b*rand([layers(1), 1]);

for i = 2:L
    W{i} = a + b*rand([layers(i), layers(i-1)]);
    B{i} = a + b*rand([layers(i), 1]);
end

% defining z and a for each layer

Z = cell(1,L);
A = cell(1,L);

% defining differentials

dW = cell(1,L);
dB = cell(1,L);
dZ = cell(1,L);
dA = cell(1,L);

% everything below should come in a for loop
cost = [];

for j = 1 : maxiter
    % forward prop

    Z{1} = W{1} * X + B{1} ;
    A{1} = g{1}(Z{1});

    for i = 2 : L
        Z{i} = W{i} * A{i-1} + B{i};
        A{i} = g{i}( Z{i} );
    end
    
%    disp(num2str(A{L}(1:5)));
    %J = - sum( (y .* log(A{L}) + (1-y) .* log(1 - A{L})) );
    J = 0;
    for k = 1 : m
        if y(k) == 1
            J = J - y(k) * log(A{L}(k));
        else
            J = J - (1 - y(k)) * log(1 - A{L}(k));
        end
    end

    cost = [cost, J];

%    dA{L} = -y ./ A{L} + (1 - y)./(1 - A{L}) ; % binary cross-entropy loss derivative
%     for k = 1 : m
%         if y(k) == 1
%             dA{L}(k) = - y(k) / A{L}(k);
%         else
%             dA{L}(k) = (1 - y(k)) / (1 - A{L}(k));
%         end
%     end

    dZ{L} = A{L} - y;
    dW{L} = dZ{L} * A{L-1}' ;
    dB{L} = sum(dZ{L}, 2) ;
    dA{L-1} = W{L}' * dZ{L} ;

    % back prop
    for i = L-1 :-1:2
        dZ{i} = dA{i} .* dg{i}( Z{i} );
        dW{i} = dZ{i} * A{i-1}' ;
        dB{i} = sum(dZ{i}, 2) ;
        dA{i-1} = W{i}' * dZ{i} ;
    end

    dZ{1} = dA{1} .* dg{1}( Z{1} );
    dW{1} = dZ{1} * X' ;
    dB{1} = sum(dZ{1}, 2) ;

    % update
    for i = 1 : L
        W{i} = W{i} - alpha * dW{i};
        B{i} = B{i} - alpha * dB{i};
    end

    disp(['iter ' num2str(j) ' Cost function: ' num2str(J)])
end

plot(1:length(cost), cost)
train_predicts = A{L} >= 0.5;
sum(train_predicts == y) / m

%% testing 

mm = 200;

X_test = zeros(M*N*3, mm);

for i = 1 : mm
    Im = imread("Cats_Test" + num2str(1700+i) + ".png");
    Im = imresize(Im, [M,N]);
    X_test(:,i) = reshape(Im, [M*N*3, 1]);
end

X_test = X_test/255;

y_test = zeros(1,mm);

for i = 1 : mm

    c = readstruct("Cats_Test" + num2str(1700+i) + ".xml");
    if c.object.name == "cat"
        y_test(i) = 1;
    else
        y_test(i) = 0;
   end
%    disp([c.object.name]);
end

% prediction by forward prop

Z{1} = W{1} * X_test + B{1} ;
A{1} = g{1}(Z{1});

for i = 2 : L
    Z{i} = W{i} * A{i-1} + B{i};
    A{i} = g{i}( Z{i} );
end

predicts = A{L} >= 0.5;
sum(predicts == y_test) / mm
