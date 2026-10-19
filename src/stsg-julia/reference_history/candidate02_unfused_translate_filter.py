"""Translate all four original STSG bodies through a typed C AST.
Requires pycparser3.00 in an isolated Python path. The emitter core was derived
from the recorded WSM5 emitter; source, types, control flow and alias support
here are bound independently and no WSM5 files are needed at runtime.
"""
from pathlib import Path
import sys,re,json,hashlib
sys.path.insert(0,'/tmp/wsm5-python-tools')
from pycparser import c_parser,c_ast
P=Path(__file__).resolve().parent;R=P.parents[1]
source=P/'native_reference_repaired_Filter.cu'
text=source.read_text(encoding='utf-8-sig')
text=re.sub(r'/\*.*?\*/','',text,flags=re.S)
text=re.sub(r'//[^\n]*','',text)
text=re.sub(r'^\s*#.*$','',text,flags=re.M)
text=text.replace('__global__','').replace('__restrict__','')
text=re.sub(r'\b(float|double|int)\s*\(',r'(\1)(',text)
(P/'filter_preprocessed.c').write_text(text)
ast=c_parser.CParser().parse(text)
# C permits omitted array lengths with a fixed initializer list. Materialize
# those compile-time dimensions before type scanning, without changing values.
def array_dimensions(n):
    if isinstance(n,c_ast.Decl) and isinstance(n.type,c_ast.ArrayDecl) and n.type.dim is None:
        assert isinstance(n.init,c_ast.InitList)
        n.type.dim=c_ast.Constant('int',str(len(n.init.exprs)))
    for _,child in n.children():array_dimensions(child)
array_dimensions(ast)
core=(P/'typed_emitter_core.py').read_text()
outputs=[];reports=[]
for fn in ast.ext:
    assert isinstance(fn,c_ast.FuncDef)
    env={'fn':fn,'c_ast':c_ast}
    exec(compile(core,'typed_emitter_core.py','exec'),env)
    base_ex=env['ex'];base_type=env['typex']
    def typex(n):
        if isinstance(n,c_ast.UnaryOp) and n.op=='&':return ('ptr',typex(n.expr))
        if isinstance(n,c_ast.UnaryOp) and n.op=='sizeof':return 'Int32'
        return base_type(n)
    def ex(n):
        if isinstance(n,c_ast.UnaryOp):
            if n.op=='&':
                assert isinstance(n.expr,c_ast.ArrayRef)
                return 'Offset('+ex(n.expr.name)+',Int32('+ex(n.expr.subscript)+'))'
            if n.op=='sizeof':
                t=typex(n.expr)
                size={'Float32':4,'Float64':8,'Int32':4,'UInt32':4,'Int16':2,'UInt8':1}
                return 'Int32('+str(size[t[1]]*t[2] if isinstance(t,tuple) else size[t])+')'
            if n.op in ['p++','p--','++','--']:
                v=ex(n.expr);one='Int32(1)';operation='+=' if '+'in n.op else '-='
                if n.op.startswith('p'):return '(('+v+' '+operation+' '+one+')'+('-'if'+'in n.op else'+')+one+')'
                return '('+v+' '+operation+' '+one+')'
        return base_ex(n)
    env['ex']=ex;env['typex']=typex
    lines=[];loop_stack=[]
    def emit(s,l):lines.append(' '*l+s)
    def stmt(n,l):
        if n is None:return
        if isinstance(n,c_ast.Compound):
            for item in n.block_items or []:stmt(item,l)
        elif isinstance(n,c_ast.Decl):
            t=env['types'][n.name]
            if isinstance(t,tuple):
                if t[0]=='ptr':
                    assert n.init is not None
                    emit('local '+n.name+' = '+ex(n.init),l)
                else:
                    values=[ex(x) for x in n.init.exprs] if isinstance(n.init,c_ast.InitList) else [(t[1]+'(NaN)')if t[1].startswith('Float')else(t[1]+'(0)')]*t[2]
                    if isinstance(n.init,c_ast.InitList):
                        values += [t[1]+'(0)'] * (t[2]-len(values))
                    emit('local '+n.name+' = MVector{'+str(t[2])+','+t[1]+'}(('+','.join(values)+',))',l)
            else:
                value=env['conv'](t,ex(n.init),typex(n.init)) if n.init is not None else(t+'(NaN)'if t.startswith('Float')else t+'(0)')
                emit('local '+n.name+'::'+t+' = '+value,l)
        elif isinstance(n,c_ast.DeclList):
            for item in n.decls:stmt(item,l)
        elif isinstance(n,c_ast.If):
            emit('if '+env['condition'](n.cond),l);stmt(n.iftrue,l+1)
            if n.iffalse is not None:emit('else',l);stmt(n.iffalse,l+1)
            emit('end',l)
        elif isinstance(n,c_ast.For):
            stmt(n.init,l);emit('while '+env['condition'](n.cond),l);loop_stack.append(('for',n.next));stmt(n.stmt,l+1);loop_stack.pop();stmt(n.next,l+1);emit('end',l)
        elif isinstance(n,c_ast.While):
            emit('while '+env['condition'](n.cond),l);loop_stack.append(('while',None));stmt(n.stmt,l+1);loop_stack.pop();emit('end',l)
        elif isinstance(n,c_ast.DoWhile):
            emit('while true',l);loop_stack.append(('do',n.cond));stmt(n.stmt,l+1);loop_stack.pop();emit('if !('+env['condition'](n.cond)+');break;end',l+1);emit('end',l)
        elif isinstance(n,c_ast.Continue):
            kind,extra=loop_stack[-1]
            if kind=='for':stmt(extra,l)
            if kind=='do':emit('if !('+env['condition'](extra)+');break;end',l)
            emit('continue',l)
        elif isinstance(n,c_ast.Break):emit('break',l)
        elif isinstance(n,c_ast.Return):assert n.expr is None;emit('return nothing',l)
        elif isinstance(n,c_ast.ExprList):
            for item in n.exprs:stmt(item,l)
        elif isinstance(n,(c_ast.Assignment,c_ast.UnaryOp,c_ast.FuncCall)):emit(ex(n),l)
        else:raise RuntimeError(('statement',type(n).__name__))
    params=[p.name for p in fn.decl.type.args.params]
    emit('function '+fn.decl.name+'_body!('+','.join(params+['bidx::UInt32','bidy::UInt32','bdimx::UInt32','bdimy::UInt32','tidx::UInt32','tidy::UInt32'])+')',0)
    emit('@inbounds begin',1);stmt(fn.body,2);emit('end',1);emit('nothing',1);emit('end',0)
    outputs.append('\n'.join(lines));reports.append({'function':fn.decl.name,'parameters':params,'locals':env['types'],'alpha_renaming':env['alpha_map']})
header='''# Full original STSG source bodies, typed C AST lowering.
# C Float32 products/double accumulation and assignment narrowing explicit.
# Fixed local arrays diagnostic NaN/0 initialized; no native undefined-value claim.
using StaticArrays
struct Offset{T}
    array::T
    offset::Int32
end
@inline Base.getindex(a::Offset,i::Integer)=@inbounds a.array[i+a.offset]
@inline function Base.setindex!(a::Offset,x,i::Integer)
    @inbounds a.array[i+a.offset]=x
    return x
end
'''
(P/'filter_device.jl').write_text(header+'\n\n'.join(outputs)+'\n')
(P/'translation_generation_review.json').write_text(json.dumps({'source':str(source.relative_to(R)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'core_sha256':hashlib.sha256((P/'typed_emitter_core.py').read_bytes()).hexdigest(),'generated_sha256':hashlib.sha256((P/'filter_device.jl').read_bytes()).hexdigest(),'pycparser_version':__import__('pycparser').__version__,'scope':'All4bodies; pointeraliases asisbitsOffset wrappers, exactlexicalshadowrename, continue/forincrement/postincrement semantics, sizeofstaticarrayfold; generatedsource unverified untilCPU+GPUchecks.','functions':reports},indent=2)+'\n')
print('Generated4functions',len((P/'filter_device.jl').read_bytes()),'bytes')
