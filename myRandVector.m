function [r]=myRandVector(N,a,b)

r = a + (b-a).*rand(N,1);

end