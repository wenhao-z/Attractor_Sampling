% Demo of a bivariate distribution

% Wen-Hao Zhang, Oct. 20, 2020

x1 = linspace(-4,4,201);
[x1, x2] = meshgrid(x1, x1);

X = [x1(:), x2(:)];
mu1 = [0, -1];
mu2 = [0, 1];
Sigma1 = [1, 0.5; ...
         0.5, 0.8];
Sigma2 = [1, 0.3; ...
         0.3, 0.8];
w1 = 0.5; 
w2 = 1 - w1;
     
y = w1 * mvnpdf(X, mu1, Sigma1) ...
    + w2 * mvnpdf(X, mu2, Sigma2);
y = reshape(y, size(x1));

%% Plot
figure(1)

% Joint distributions
contourf(x1, x2, y, 'levelstep', 5e-3, 'linestyle', 'none')

cMap = getColorMapPosNegDat([0, max(y(:))], 64);
colormap(cMap);
colorbar
% cmap = hot(64);
% colormap(flipud(cmap))

axis square
grid on

% Marginal distributions
figure(2)
hAxe(1) = subplot(2,1,1);
y1 = mean(y,1);
plot(x1(1,:), y1/sum(y1))
xlabel('s_1')
% axis square
box off

hAxe(2) = subplot(2,1,2);
y2 = mean(y,2);
plot(x2(:,1), y2/sum(y2));
% axis square
xlabel('s_2')
box off

set(hAxe, 'ylim', [0, 2e-2])
linkaxes(hAxe, 'y')
