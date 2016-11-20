! -*- f90 -*-
! file lbfgs.f90
! Approximate Hessian inversion
! version 0.1
PROGRAM test_lbfgs
    IMPLICIT NONE
    
    INTEGER row, col, i
    REAL (KIND=8) :: x(3), d(3), H_0(3,3),s(3,10), y(3,10)   
    REAL (KIND=8) :: f!, g_f

    do row=1,3
        do col=1,3
            if(ABS(row-col)<0.1) then
                H_0(row,col)=1.0
            else
                H_0(row,col)=0
            end if
        end do
        x(row) = 60/row 
    end do 

    do row=1,3
        do col=1,10
            s(row, col) = 0
            y(row, col) = 0
        end do
    end do


    print *,'x',x
    !print *,f(x,3)

    do i = 1,10
        d = 2*x+2*(/1,2,3/)!g_f(x,3)

        CALL lbfgs(3, H_0, i, s, y, d)

        print *,'d',d

        if (DOT_PRODUCT(d,d) < 0.0001 ) then
            EXIT
        endif

        x = x-d

        s(:,i) = -1*d
        y(:,i) = -2*d

        print *,'x',x
        print *,'f(x)',f(x,3)
    end do

END PROGRAM test_lbfgs

FUNCTION f(x, n)
    IMPLICIT NONE

    INTEGER :: n,i
    REAL (KIND=8):: x(n), f
    
    f = 0.0

    do i=1,n
        f = f+(i+x(i))**2
    end do

    f=f

END FUNCTION f

FUNCTION g_f(x,n)

    IMPLICIT NONE
    INTEGER N
    REAL (KIND=8) :: x(n), g_f(n)

    g_f = 2*x

END FUNCTION g_f

SUBROUTINE lbfgs(n, H_0, k, s, y, d)

!*****************************************************************
!
!   subroutine lbfgs
!   
!   Approximately apply the inverse Hessian to d.
!   m: dim of H_0 (mxm) and d (m)
!   d: Vector to mupltiply by. On exit, the result of the multiplication
!
!*****************************************************************
    IMPLICIT NONE
    
    INTEGER ::  k, n
    REAL (KIND=8) ::  H_0(n,n), s(n,k), y(n,k), d(n) 

    INTEGER :: i,j
    REAL (KIND=8) ::  a(k), b(k), rho(k) 


    do i=1,k-1
        !print *, y(:,i), s(:,i)
        !print *,1/DOT_PRODUCT(y(:,i), s(:,i))
        rho(i) = 1.0/DOT_PRODUCT(y(:,i), s(:,i))
    end do

    do i=k-1, 1,-1
        a(i) =  rho(i)*(DOT_PRODUCT(s(:,i), d))
        !print *,i,a(i), rho(i)
        d = d - a(i)*y(:,i) ! might need to modify here?
        !print *,'d',d
    end do

    !print *,'d',d
    !print *,'H0', H_0

    d = MATMUL(H_0, d)
    !print *,'d',d

    do i=1,k-1
       b(i) = rho(i)*(DOT_PRODUCT(y(:,i), d)) 
       !print *,i, b(i), rho(i)
       d = d + (a(i)-b(i))*s(:,i) ! might also need a tweak here
       !print *,'d',d
    end do

END SUBROUTINE lbfgs
