
%%
% Run demoCANNSampling.m first

figure;
hold on
x = -180:1:180;
a = 40;
s = -180:45:180;
cSpec = flipud(autumn(length(s)));
for iter = 1:length(s)
    y = normpdf(x, s(iter), a);
    plot(x,y, 'color', cSpec(iter,:))
end
set(gca, 'xlim', [-180, 180], 'xtick',-180:90:180)

rate = squeeze(InputSet.O(:,1,1e4));

plot(NetPars.PrefStim, rate/(sum(rate)*mean(diff(NetPars.PrefStim))), 'b');
% set(gca, 'xlim', [-180, 180], 'xtick',-180:90:180)