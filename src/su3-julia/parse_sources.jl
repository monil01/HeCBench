function inspect(expr)
 if expr isa Expr
  expr.head in (:error,:incomplete)&&error(expr)
  foreach(inspect,expr.args)
 end
end
for path in ARGS
 inspect(Meta.parseall(read(path,String);filename=path))
 println("Parsed ",path)
end
