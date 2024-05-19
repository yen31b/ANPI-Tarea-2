function parte1_p2_oct
  clc; clear;
  %parametros
  x0 = zeros(242,1);
  p=q=[1:0.1:25.1];
  m=242;
  b=ones(m,1);
  tol=1e-5;
  iterMax=1000;
  A=tridiagonal(p,q,m);
  %llamada al metodo de jacobi con los parametros definidos
  xk = metodo_jacobi(A,b,x0,tol,@criterio_parada,iterMax)

end

function A = tridiagonal(p, q, m)
  % Crear la matriz A
  A = zeros(m);
  % Definir los valores que no varian en la matriz
  A(1,1) = 2*q(1);
  A(1,2) = q(1);
  A(2,1)= p(2);
  A(2,3)= q(2);
  % definir los valores que varian por el indice
  for i = 2:m-1
    A(i,i-1) = p(i); %diagonal inferior
    A(i,i) = 2*(p(i) + q(i)); %diagonal principal
    A(i,i+1) = q(i); %diagonal superior
  endfor
  % Definir valores finales de la matriz
  A(m,m-1) = p(m);
  A(m,m) = 2*p(m);
end

function xk = metodo_jacobi(A,b,x0,tol,criterio_parada, iterMax)
  %obtener elementos de la diagonal
  d=diag(A);
  %dimension de b para saber cual debe ser la dim de aproximacion xk
  m=length(b);
  %inicializar xk
  xk=x0;
  % Calcular la aproximacion por medio de la formula
  for k=1:iterMax
    xk1= zeros(m,1);
    for i=1:m
      sumatoria=0;
      for j=1:m
        if j != i
          sumatoria = sumatoria + A(i, j) * xk(j);
        endif
      endfor
      xk1(i) = (b(i) - sumatoria) / A(i, i);
    endfor
    criterio_parada = norm(A*xk1-b);
    if criterio_parada<tol
      break
    endif
    xk=xk1;
  endfor
end
