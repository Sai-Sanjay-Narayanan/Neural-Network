function y = relu(x)
    y = ( sign(x) + 1 ) .* x / 2;
    %y = ( sign(x) + 1 ) .* x / 2 + 0.01 * (1 - sign(x)) .* x / 2;
end