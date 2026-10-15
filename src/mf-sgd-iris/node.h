#pragma once
struct mf_node { int u,v; float rate; };
static_assert(sizeof(mf_node)==12,"source binary rating record layout");
