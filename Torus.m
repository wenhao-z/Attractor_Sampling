clc
close all
clear all
R=2;
r=1;


u=linspace(0,2*pi,30);
v=linspace(0,2*pi,30);
[u,v]=meshgrid(u,v);

x=(R+r.*cos(v)).*cos(u);
y=(R+r.*cos(v)).*sin(u);
z=r.*sin(v);

figure; 
hold on

mesh(x,y,z);
view([0,65])

% Plot two circles
u=linspace(0,2*pi,100);
v=pi/2;
[u,v]=meshgrid(u,v);

x=(R+r.*cos(v)).*cos(u);
y=(R+r.*cos(v)).*sin(u);
z=r.*sin(v);
plot3(x,y,z,'g')

u=0;
v=linspace(0,2*pi,100);
[u,v]=meshgrid(u,v);

x=(R+r.*cos(v)).*cos(u);
y=(R+r.*cos(v)).*sin(u);
z=r.*sin(v);
plot3(x,y,z,'r')




% h=gca; 
% get(h,'FontSize') 
% set(h,'FontSize',14)
% xlabel('X','fontSize',14);
% ylabel('Y','fontSize',14);
% zlabel('Z','fontsize',14);
% title('Torus','fontsize',14)
% fh = figure(1);
% set(fh, 'color', 'white'); 

%%

%%Create R and THETA data
theta = 0:pi/10:2*pi;
r = 2*pi:pi/20:3*pi;
[R,T] = meshgrid(r,theta);
%%Create top and bottom halves
Z_top = 2*sin(R);
Z_bottom = -2*sin(R);
%%Convert to Cartesian coordinates and plot
[X,Y,Z] = pol2cart(T,R,Z_top);
surf(X,Y,Z);
hold on;
[X,Y,Z] = pol2cart(T,R,Z_bottom);
m(X,Y,Z);
axis equal
shading interp