function [cmap, cIdxZero, colortick]= getColorMapPosNegDat(cAxis, cLength)

if nargin < 2
   cLength = 64; 
end

colorLim = cAxis;
colortick = linspace(colorLim(1), colorLim(2), cLength);

cSpec = lines(2);
% cSpec = cool(2);
% cSpec = jet(2);

cSpec = flip(cSpec, 1);
% cSpec = [1,0,0; ...
%     0,0,1];cSpec = lines(2);
cSpec = cSpec(2,:);
cMap = linspace(cSpec, 1, 64);
cMap = ones(1,3);

[~, cIdxZero] = min(abs(colortick));

cmap = zeros(cLength, 3);

cmap(cIdxZero,:) = ones(1,3);
for iter = 1:3
    cmap(1:cIdxZero, iter) = linspace(cSpec(1,iter), 1, cIdxZero);
    cmap(cIdxZero:end, iter) = linspace(1, cSpec(2,iter), cLength+1-cIdxZero);
end
