% Plot a visual contour embedded in line segments, similar with Wu Li's
% experiment

% Wen-Hao Zhang, Oct. 20, 2020

N = 13; % size of grid

% Generate the end point of every line segment
r = 1; % The length of every line segment
theta = pi *rand(1, N^2);

[xGrid, yGrid] = ind2sub([N,N], 1:N^2);

X = [xGrid - cos(theta)/2; ...
    xGrid + cos(theta)/2];

Y = [yGrid - sin(theta)/2; ...
    yGrid + sin(theta)/2];

%%
cSpec = lines(1);
plot(X,Y, 'color', cSpec);
axis square