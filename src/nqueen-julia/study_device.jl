using StaticArrays
struct QueenRoot
 control::UInt32
 board::NTuple{12,Int8}
end
@inline function queens_legal(board,r::Int32)
 safe=true
 @inbounds begin
  for i in Int32(0):r-Int32(1)
   if board[i+1]==board[r+1];safe=false;end
  end
  ld=Int32(board[r+1]);rd=ld
  for i in r-Int32(1):Int32(-1):Int32(0)
   ld-=Int32(1);rd+=Int32(1)
   if board[i+1]==ld||board[i+1]==rd;safe=false;end
  end
 end
 safe
end
@inline function queen_dfs(root,N::Int32,D::Int32)
 board=MVector{20,Int8}(undef)
 @inbounds begin
  for i in Int32(1):N;board[i]=Int8(-1);end
  for i in Int32(1):D;board[i]=root.board[i];end
  flag=root.control;depth=D;tree=UInt64(0);solutions=UInt64(0)
  while true
   board[depth+1]+=Int8(1);column=Int32(board[depth+1]);bit=UInt32(1)<<column
   if column==N
    board[depth+1]=Int8(-1)
   elseif (flag&bit)==0&&queens_legal(board,depth)
    tree+=UInt64(1);flag|=bit;depth+=Int32(1)
    if depth==N;solutions+=UInt64(1);else;continue;end
   else
    continue
   end
   depth-=Int32(1);flag&=~(UInt32(1)<<Int32(board[depth+1]))
   if depth<D;break;end
  end
 end
 tree,solutions
end
