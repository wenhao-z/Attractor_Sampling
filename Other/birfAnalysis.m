% Numerical analysis of bifurcations
% Wen-Hao Zhang


g12 = -0.05;
g21 = 0.5;
h = 1;
dt = 0.01;

M = [h + g12, -g12;  - g21, g21];
eig(M)

z = zeros(2, 1e2);
z(:,1) = rand(1)*ones(2,1);
% z(:,1) = rand(2,1);

eps = 1e-10;
dz = 1;
t = 1;
% while norm(dz) > eps
while t < 1e4
    dz =  -M*dt * z(:,t);
    z(:,t+1) =  z(:,t) + dz;

    t = t+ 1;
end

phase = z(1,:) + 1i * z(2,:);
phase = angle(phase);

%%

subplot(2,2,1)
hold on
plot(1:t, z(1,1:t));
plot(1:t, z(2,1:t));
xlabel('Time t')
axis square

subplot(2,2,2)
hold on
plot(1:t-1, diff(z(1,1:t)));
plot(1:t-1, diff(z(2,1:t)));
xlabel('Time t')
ylabel('dz')
axis square

subplot(2,2,3)
plot(1:t,phase)
xlabel('Time t')
ylabel('phase')
axis square 

subplot(2,2,4)
plot(z(1,:), z(2,:))
xlabel('z_1')
ylabel('z_2')
axis square
