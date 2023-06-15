% Im = imread("Cats_Test1.png");
% Im = imresize(Im, [64,64]);
% imshow(Im(:,:,1))
m = 1500;
M = 64;
N = 64;
X = zeros(M*N*3, m);

for i = 1 : m
    Im = imread("Cats_Test" + num2str(i) + ".png");
    Im = imresize(Im, [M,N]);
    X(:,i) = reshape(Im, [M*N*3, 1]);
end

X = X/255;
%%
y = zeros(1,m);

% xmlfile = 'Cats_Test736.xml';
% c = readstruct(xmlfile)
% c.object.name

for i = 1 : m

    c = readstruct("Cats_Test" + num2str(i) + ".xml");
    if c.object.name == "cat"
        y(i) = 1;
    else
        y(i) = 0;
   end
%    disp([c.object.name]);
end