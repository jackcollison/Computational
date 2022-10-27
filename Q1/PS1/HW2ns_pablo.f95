!     Last change:  PND   6 Sep 2007    5:56 pm

module parameters
implicit none
   REAL, PARAMETER  	:: b = 0.99, d = 0.025, a = 0.36
   REAL, PARAMETER  	:: klb = 0.01, inc = 0.025, kub = 45.0
   REAL, PARAMETER      :: HH = 0.977, HL = 0.023, LH = 0.074, LL = 0.926
   REAL, PARAMETER      :: Zg = 1.25, Zb = 0.2
   INTEGER, PARAMETER 	:: length_grid_k = (kub-klb)/inc + 1
   REAL , PARAMETER 	:: toler   = 1.e-4						! Numerical tolerance
end module

! ============================================================================================
module global
USE parameters
implicit none
   REAL 		:: Kgrid(length_grid_k), value(length_grid_k, 2), g_k(length_grid_k, 2), Zgrid(2), P(2, 2)
   REAL                 :: vtmp(length_grid_k * 2,length_grid_k), value_new(length_grid_k, 2)
end module

! ============================================================================================

PROGRAM  HW2Stochastic
   REAL 			:: total, etime, dist
   REAL, DIMENSION(2)  		:: elapsed

   call solution

   total=etime(elapsed)


	PRINT*,'--------------------------------------------------'
	PRINT*,'total time elpased =',total
	PRINT*,'--------------------------------------------------'


END PROGRAM HW2Stochastic

! ============================================================================================
subroutine solution
USE parameters
USE global

   IMPLICIT  NONE

   INTEGER :: iter, index_z, index_k, index_kp
   REAL :: diff, k, kp, c, z
   
   INTEGER :: i = 1


   do while (i<=length_grid_k)   !do loop for assigning capital grid K
     Kgrid(i) = klb + (i-1)*inc
     i = i + 1
   end do

   Zgrid(1) = Zg
   Zgrid(2) = Zb

   P(1, 1) = HH
   P(1, 2) = HL
   P(2, 1) = LH
   P(2, 2) = LL

   iter = 1
   diff = 1000.d0
   value(:,1) = 0.*Kgrid		!Initial Value guess
   value(:,2) = 0.*Kgrid
   
	do while (diff>= toler)
                do index_z = 1, 2
                        z = Zgrid(index_z)
                        do index_k = 1, length_grid_k				! Capital grid
                                k = Kgrid(index_k)
                                vtmp(index_k * index_z,:) = -1.0e-16

                                do index_kp = 1, length_grid_k
                                        kp = Kgrid(index_kp)
                                        c = z * k**a + (1.-d)*k - kp

                                        if (c>0.) then
                                                vtmp(index_k * index_z, index_kp) = log(c) + b * &
                                                 (value(index_kp, 1) * P(index_z, 1) + value(index_kp, 2) * P(index_z, 2)) 
                                        endif

                                enddo

                                value_new(index_k, index_z) = MAXVAL(vtmp(index_k * index_z,:))
                                g_k(index_k, index_z) 	    = Kgrid(MAXLOC(vtmp(index_k * index_z,:),1))
                        enddo
                enddo

		diff  = MAX(maxval(abs(value_new(:, 1)-value(:, 1))) / ABS(value_new(length_grid_k ,1)), &
                maxval(abs(value_new(:, 2)-value(:, 2))) / ABS(value_new(length_grid_k ,2)))
		value = value_new


		! print*, 'Iteration =',iter,'sup_norm =',diff
                iter = iter+1

	enddo

	print *, ' '
	print *, 'Successfully converged with sup_norm ', diff
    	!print *, g_k
    
    !CALL vcDrawCurve@(d, Kgrid, g_k, length_grid_k)


        open (UNIT=1,FILE='valuefun',STATUS='replace')
	do index_k = 1, length_grid_k
        	WRITE(UNIT=1,FMT=*) value(index_k, 1), ",", value(index_k, 2)
        end do
	close (UNIT=1)

        open (UNIT=1,FILE='policyfun',STATUS='replace')
	do index_k = 1, length_grid_k
        	WRITE(UNIT=1,FMT=*) g_k(index_k, 1), ",", g_k(index_k, 2)
        end do
	close (UNIT=1)

        open (UNIT=1,FILE='diff',STATUS='replace')
	do index_k = 1, length_grid_k
        	WRITE(UNIT=1,FMT=*) g_k(index_k, 1) - Kgrid(index_k), ",", g_k(index_k, 2) - Kgrid(index_k)
        end do
	close (UNIT=1)

end subroutine
